// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - Constants

private let forcedPrefix: [Int32] = [50258, 50259, 50360, 50364]  // BOS, <|en|>, <|transcribe|>, <|notimestamps|>
private let eotToken: Int32 = 50257
private let maxTargetPositions = 448
private let maxDecodeSteps = 50
private let melElements = 128 * 3000  // (128, 3000) flattened

// MARK: - Entry point

@main
struct Main {
    static func main() async {
        // Usage: whisper-runner [bundle-dir] [mel-bin]
        let bundleDir = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1]
            : "/tmp/whisper-work/exports/whisper-large-v3-turbo_float32_coreai"
        let melBinPath = CommandLine.arguments.count > 2
            ? CommandLine.arguments[2]
            : nil
        do {
            try await runWhisper(bundleDir: bundleDir, melBinPath: melBinPath)
        } catch {
            print("Fatal: \(error)")
            exit(1)
        }
    }
}

// MARK: - Mel loading

/// Load a raw float32 mel binary saved by compute_mel.py.
/// Shape on disk: (128, 3000) = 384000 floats. We wrap it as (1, 128, 3000).
private func loadMel(from path: String, descriptor: NDArrayDescriptor) throws -> NDArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let count = data.count / MemoryLayout<Float>.size
    guard count == melElements else {
        fatalError("mel bin has \(count) floats, expected \(melElements)")
    }
    var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, 128, 3000]))
    let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    fillNDArray(&array, as: Float.self, with: floats)
    return array
}

// MARK: - Runner

func runWhisper(bundleDir: String, melBinPath: String?) async throws {
    let encURL = URL(fileURLWithPath: "\(bundleDir)/encoder.aimodel")
    let decURL = URL(fileURLWithPath: "\(bundleDir)/decoder.aimodel")

    // ── Load models ──────────────────────────────────────────────────────────

    print("Loading encoder…")
    let encModel = try await AIModel(contentsOf: encURL)
    print("  functions: \(encModel.functionNames)")

    print("Loading decoder…")
    let decModel = try await AIModel(contentsOf: decURL)
    print("  functions: \(decModel.functionNames)")

    // Device placement: chunkedStatic → ANE, dynamic → GPU
    let encOnANE = encModel.functionNames.contains { $0.hasPrefix("extend") }
    let decOnANE = decModel.functionNames.contains { $0.hasPrefix("extend") }
    print("\nDevice placement:")
    print("  encoder → \(encOnANE ? "ANE" : "GPU")")
    print("  decoder → \(decOnANE ? "ANE" : "GPU")")

    guard let encFn = try encModel.loadFunction(named: "main"),
          let decFn = try decModel.loadFunction(named: "main")
    else { fatalError("No 'main' function in model") }

    let encDesc = encModel.functionDescriptor(for: "main")!
    let decDesc = decModel.functionDescriptor(for: "main")!

    print("\nEncoder I/O: inputs=\(encDesc.inputNames)  outputs=\(encDesc.outputNames)")
    print("Decoder I/O: inputs=\(decDesc.inputNames)  states=\(decDesc.stateNames)  outputs=\(decDesc.outputNames)")

    // ── Encoder ──────────────────────────────────────────────────────────────

    guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features"),
          case .ndArray(let encOutNDDesc) = encDesc.outputDescriptor(of: "encoder_hidden_states")
    else { fatalError("Unexpected encoder descriptor kinds") }

    let encOutShape = encOutNDDesc.shape  // [1, 1500, 1280]

    // Load mel: from file if provided, otherwise silence (zeros)
    var melArray: NDArray
    if let path = melBinPath {
        print("\nLoading mel from \(path)…")
        melArray = try loadMel(from: path, descriptor: melNDDesc)
    } else {
        print("\nUsing silence (dummy mel)…")
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillNDArray(&melArray, as: Float.self, count: melElements) { _ in 0.0 }
    }

    var encOutArray = NDArray(descriptor: encOutNDDesc.resolvingDynamicDimensions(encOutShape))

    // Warmup
    do {
        var out = InferenceFunction.MutableViews()
        out.insert(&encOutArray, for: "encoder_hidden_states")
        _ = try await encFn.run(
            inputs: ["input_features": melArray],
            states: InferenceFunction.MutableViews(),
            outputViews: consume out
        )
    }

    // Timed
    print("\n── Encoder ─────────────────────────────────────────────────────────────")
    let encT0 = Date()
    do {
        var out = InferenceFunction.MutableViews()
        out.insert(&encOutArray, for: "encoder_hidden_states")
        _ = try await encFn.run(
            inputs: ["input_features": melArray],
            states: InferenceFunction.MutableViews(),
            outputViews: consume out
        )
    }
    print(String(format: "  latency: %.1f ms", Date().timeIntervalSince(encT0) * 1000))

    // ── Decoder setup ─────────────────────────────────────────────────────────

    guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
          case .ndArray(let posIdsNDDesc)   = decDesc.inputDescriptor(of: "position_ids"),
          case .ndArray(let encHSNDDesc)    = decDesc.inputDescriptor(of: "encoder_hidden_states"),
          case .ndArray(let keyCacheNDDesc) = decDesc.stateDescriptor(of: "keyCache"),
          case .ndArray(let valCacheNDDesc) = decDesc.stateDescriptor(of: "valueCache"),
          case .ndArray(let logitsNDDesc)   = decDesc.outputDescriptor(of: "logits")
    else { fatalError("Unexpected decoder descriptor kinds") }

    let vocabSize = logitsNDDesc.shape.last!
    let kcShape = keyCacheNDDesc.shape.map { $0 < 0 ? maxTargetPositions : $0 }
    let vcShape = valCacheNDDesc.shape.map { $0 < 0 ? maxTargetPositions : $0 }
    var keyCache   = NDArray(descriptor: keyCacheNDDesc.resolvingDynamicDimensions(kcShape))
    var valueCache = NDArray(descriptor: valCacheNDDesc.resolvingDynamicDimensions(vcShape))
    print("\n── Decoder ─────────────────────────────────────────────────────────────")
    print("  KV cache shape: \(kcShape)  vocabSize: \(vocabSize)")

    // Copy encoder output into decoder's encoder_hidden_states input
    let encFlat = readNDArray(encOutArray, as: Float.self, count: encOutShape.reduce(1, *))
    var encHSArray = NDArray(descriptor: encHSNDDesc.resolvingDynamicDimensions(encOutShape))
    fillNDArray(&encHSArray, as: Float.self, with: encFlat)

    var logitsArray = NDArray(descriptor: logitsNDDesc.resolvingDynamicDimensions([1, 1, vocabSize]))

    // ── Prime KV cache with forced prefix (untimed) ───────────────────────────

    var tokens: [Int32] = forcedPrefix
    var processedCount = 0

    for tok in forcedPrefix {
        let seqLen = processedCount + 1
        var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        var pos = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, seqLen]))
        fillNDArray(&ids, as: Int32.self, with: [tok])
        fillNDArray(&pos, as: Int32.self, count: seqLen) { Int32($0) }
        var st = InferenceFunction.MutableViews()
        st.insert(&keyCache, for: "keyCache")
        st.insert(&valueCache, for: "valueCache")
        var out = InferenceFunction.MutableViews()
        out.insert(&logitsArray, for: "logits")
        _ = try await decFn.run(
            inputs: ["input_ids": ids, "position_ids": pos, "encoder_hidden_states": encHSArray],
            states: consume st, outputViews: consume out
        )
        processedCount += 1
    }

    // ── Greedy decode (timed) ─────────────────────────────────────────────────

    print("  decoding (max \(maxDecodeSteps) steps)…")
    var stepTimesMs: [Double] = []

    while stepTimesMs.count < maxDecodeSteps {
        let seqLen = processedCount + 1
        var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        var pos = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, seqLen]))
        fillNDArray(&ids, as: Int32.self, with: [tokens.last!])
        fillNDArray(&pos, as: Int32.self, count: seqLen) { Int32($0) }
        var st = InferenceFunction.MutableViews()
        st.insert(&keyCache, for: "keyCache")
        st.insert(&valueCache, for: "valueCache")
        var out = InferenceFunction.MutableViews()
        out.insert(&logitsArray, for: "logits")

        let t0 = Date()
        _ = try await decFn.run(
            inputs: ["input_ids": ids, "position_ids": pos, "encoder_hidden_states": encHSArray],
            states: consume st, outputViews: consume out
        )
        stepTimesMs.append(Date().timeIntervalSince(t0) * 1000)

        let logits = flattenAsFloat(logitsArray)
        let nextTok = Int32(logits.indices.max(by: { logits[$0] < logits[$1] })!)
        tokens.append(nextTok)
        processedCount += 1
        if nextTok == eotToken { break }
    }

    // ── Results ───────────────────────────────────────────────────────────────

    let avgMs = stepTimesMs.reduce(0, +) / Double(stepTimesMs.count)
    print(String(format: "  steps:            %d", stepTimesMs.count))
    print(String(format: "  avg step latency: %.1f ms", avgMs))
    print(String(format: "  throughput:       %.1f tok/s", 1000.0 / avgMs))
    if let lo = stepTimesMs.min(), let hi = stepTimesMs.max() {
        print(String(format: "  min/max:          %.1f / %.1f ms", lo, hi))
    }
    print("  token ids: \(tokens)")

    // ── Decode tokens to text ─────────────────────────────────────────────────

    print("\n── Transcription ────────────────────────────────────────────────────────")
    do {
        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v3-turbo")
        // skip_special_tokens: filter out ids >= 50257 (special token range for Whisper)
        let contentIds = tokens.filter { $0 < 50257 }.map { Int($0) }
        let text = tokenizer.decode(tokens: contentIds)
        print("  \(text)")
    } catch {
        print("  (tokenizer error: \(error))")
        print("  token ids: \(tokens)")
    }
}
