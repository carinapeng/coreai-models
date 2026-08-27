// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

/// End-to-end block-diffusion runner for DiffusionGemma bundles.
///
/// Loads the `encoder.aimodel` and `decoder.aimodel` components by path (the
/// diffusion bundle schema is not understood by the standard `LanguageBundle`
/// loader), runs the encoder prefill to fill the KV cache, then drives the
/// bidirectional decoder through the reference diffusion loop: random-init
/// canvas, entropy-bound accept/renoise, self-conditioning, a linear temperature
/// schedule, and a stable+confident stop.
enum DiffusionGemmaRunner {
    static let tMax: Float = 0.8
    static let tMin: Float = 0.4
    static let entropyBound: Float = 0.1
    static let confidenceThreshold: Float = 0.005
    static let bosToken = 2
    static let hiddenSize = 2816
    // Unified cache geometry baked into the exported encoder.
    static let cacheLayers = 30
    static let cacheKVHeads = 8
    static let cacheHeadDim = 512

    static func chatFormat(_ prompt: String) -> String {
        "<|turn>user\n\(prompt)<turn|>\n<|turn>model\n"
    }

    static func run(
        bundleDir: String,
        prompt: String,
        canvasLength: Int,
        maxSteps: Int
    ) async throws {
        let dir = URL(fileURLWithPath: bundleDir, isDirectory: true)
        setvbuf(stdout, nil, _IONBF, 0)  // unbuffered so progress survives a crash
        let encURL = dir.appendingPathComponent("encoder.aimodel")
        let decURL = dir.appendingPathComponent("decoder.aimodel")
        let tokURL = dir.appendingPathComponent("tokenizer")
        print("[dg] loading tokenizer from \(tokURL.path)")

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokURL)
        let ids = [bosToken] + tokenizer.encode(text: chatFormat(prompt), addSpecialTokens: false)
        let encLen = ids.count
        print("prompt ids (\(encLen)): \(ids)")

        let encFn = try await loadMainFunction(at: encURL)
        print("[dg] encoder loaded")
        let decFn = try await loadMainFunction(at: decURL)
        print("[dg] decoder loaded")

        // ---- 1. Encoder prefill (functional cache: zeros in, filled out). ----
        // The static encoder is functional: keyCache/valueCache are INPUTS (zeros
        // in) and OUTPUTS (filled out), not persistent state. Cache seq == encLen.
        let encDesc = encFn.descriptor
        func encIn(_ name: String) throws -> NDArrayDescriptor {
            guard case .ndArray(let d) = encDesc.inputDescriptor(of: name) else {
                throw DiffusionRunnerError.contract("encoder missing input '\(name)'")
            }
            return d
        }
        let cacheShape = [cacheLayers, 1, cacheKVHeads, encLen, cacheHeadDim]
        let cacheCount = cacheLayers * cacheKVHeads * encLen * cacheHeadDim
        let zeros = [Float](repeating: 0, count: cacheCount)

        var encInIds = NDArray(descriptor: try encIn("input_ids").resolvingDynamicDimensions([1, encLen]))
        fillNDArray(&encInIds, as: Int32.self, count: encLen) { Int32(ids[$0]) }
        var encPos = NDArray(descriptor: try encIn("position_ids").resolvingDynamicDimensions([1, encLen]))
        fillNDArray(&encPos, as: Int32.self, count: encLen) { Int32($0) }
        var keyIn = NDArray(descriptor: try encIn("keyCache").resolvingDynamicDimensions(cacheShape))
        fillFloatNDArray(&keyIn, with: zeros)
        var valueIn = NDArray(descriptor: try encIn("valueCache").resolvingDynamicDimensions(cacheShape))
        fillFloatNDArray(&valueIn, with: zeros)
        print("[dg] encoder inputs built, running prefill...")

        var encOutputs = try await encFn.run(
            inputs: [
                "input_ids": encInIds, "position_ids": encPos,
                "keyCache": keyIn, "valueCache": valueIn,
            ],
            outputViews: InferenceFunction.MutableViews()
        )
        guard let keyOut = encOutputs.remove("keyCache")?.ndArray,
            let valueOut = encOutputs.remove("valueCache")?.ndArray
        else {
            throw DiffusionRunnerError.contract("encoder missing keyCache/valueCache output")
        }
        print("[dg] encoder prefill done")

        // Filled encoder prefix [L,1,heads,encLen,headDim] for decoder cross-attention.
        let encoderK = flattenAsFloat(keyOut)
        let encoderV = flattenAsFloat(valueOut)

        // ---- 2. Diffusion loop over one canvas. ------------------------------
        let decDesc = decFn.descriptor
        let vocab = try decoderVocabSize(decDesc)
        var canvas = (0..<canvasLength).map { _ in Int32.random(in: 0..<Int32(vocab)) }
        var soft = [Float](repeating: 0, count: canvasLength * hiddenSize)

        var prevArgmax: [Int32]? = nil
        var argmaxCanvas = canvas
        var bestArgmax = canvas
        var bestMeanH = Float.greatestFiniteMagnitude
        for step in stride(from: maxSteps, through: 1, by: -1) {
            let temp = tMin + (tMax - tMin) * (Float(step) / Float(maxSteps))
            let (processed, softOut) = try await runDecoder(
                decFn, decDesc, canvas: canvas, soft: soft, encLen: encLen,
                canvasLength: canvasLength, vocab: vocab,
                encoderK: encoderK, encoderV: encoderV, temperature: temp)
            soft = softOut

            argmaxCanvas = (0..<canvasLength).map { row in
                argmaxRow(processed, row: row, vocab: vocab)
            }
            let entropies = (0..<canvasLength).map { entropyRow(processed, row: $0, vocab: vocab) }

            // Entropy-bound accept: lowest-entropy positions whose cumulative
            // (excluding self) entropy stays within the bound.
            let order = (0..<canvasLength).sorted { entropies[$0] < entropies[$1] }
            var accept = [Bool](repeating: false, count: canvasLength)
            var cum: Float = 0
            for idx in order {
                if cum <= entropyBound { accept[idx] = true }
                cum += entropies[idx]
            }

            var next = [Int32](repeating: 0, count: canvasLength)
            for i in 0..<canvasLength {
                if accept[i] {
                    next[i] = sampleRow(processed, row: i, vocab: vocab)
                } else {
                    next[i] = Int32.random(in: 0..<Int32(vocab))
                }
            }
            canvas = next

            let meanH = entropies.reduce(0, +) / Float(canvasLength)
            if meanH < bestMeanH {
                bestMeanH = meanH
                bestArgmax = argmaxCanvas
            }
            let acceptedCount = accept.filter { $0 }.count
            let preview = tokenizer.decode(tokens: argmaxCanvas.map { Int($0) }.filter { $0 != 0 })
            print(
                "  step \(step) temp=\(String(format: "%.3f", temp)) "
                    + "meanH=\(String(format: "%.3f", meanH)) accept=\(acceptedCount)/\(canvasLength) "
                    + "| \(preview.replacingOccurrences(of: "\n", with: " ").prefix(80))")

            let stable = prevArgmax.map { $0 == argmaxCanvas } ?? false
            prevArgmax = argmaxCanvas
            if stable && meanH < confidenceThreshold {
                print("stop at step \(step)")
                break
            }
        }

        // Trim at the first end-of-turn / eos token for a clean answer.
        var finalIds: [Int] = []
        for t in bestArgmax {
            if t == 1 || t == 106 { break }
            if t != 0 { finalIds.append(Int(t)) }
        }
        let text = tokenizer.decode(tokens: finalIds)
        print("\n=== block diffusion (Swift llm-runner) ===\ntext: \(text)")
    }

    // MARK: - Decoder step

    private static func runDecoder(
        _ fn: InferenceFunction,
        _ desc: InferenceFunctionDescriptor,
        canvas: [Int32],
        soft: [Float],
        encLen: Int,
        canvasLength: Int,
        vocab: Int,
        encoderK: [Float],
        encoderV: [Float],
        temperature: Float
    ) async throws -> (processed: [Float], soft: [Float]) {
        func inDesc(_ name: String) throws -> NDArrayDescriptor {
            guard case .ndArray(let d) = desc.inputDescriptor(of: name) else {
                throw DiffusionRunnerError.contract("decoder missing input '\(name)'")
            }
            return d
        }
        var dIds = NDArray(descriptor: try inDesc("decoder_input_ids").resolvingDynamicDimensions([1, canvasLength]))
        fillNDArray(&dIds, as: Int32.self, count: canvasLength) { canvas[$0] }

        var prevSoft = NDArray(
            descriptor: try inDesc("prev_soft_embeds").resolvingDynamicDimensions([1, canvasLength, hiddenSize]))
        fillFloatNDArray(&prevSoft, with: soft)

        var pos = NDArray(descriptor: try inDesc("position_ids").resolvingDynamicDimensions([1, canvasLength]))
        fillNDArray(&pos, as: Int32.self, count: canvasLength) { Int32(encLen + $0) }

        let ekShape = [cacheLayers, 1, cacheKVHeads, encLen, cacheHeadDim]
        var ek = NDArray(descriptor: try inDesc("encoder_k").resolvingDynamicDimensions(ekShape))
        fillFloatNDArray(&ek, with: encoderK)
        var ev = NDArray(descriptor: try inDesc("encoder_v").resolvingDynamicDimensions(ekShape))
        fillFloatNDArray(&ev, with: encoderV)

        var tempArr = NDArray(descriptor: try inDesc("temperature").resolvingDynamicDimensions([1]))
        fillFloatNDArray(&tempArr, with: [temperature])

        var outputs = try await fn.run(
            inputs: [
                "decoder_input_ids": dIds, "prev_soft_embeds": prevSoft, "position_ids": pos,
                "encoder_k": ek, "encoder_v": ev, "temperature": tempArr,
            ],
            outputViews: InferenceFunction.MutableViews()
        )
        guard let logits = outputs.remove("logits")?.ndArray,
            let softOut = outputs.remove("soft_embeds")?.ndArray
        else {
            throw DiffusionRunnerError.contract("decoder missing logits/soft_embeds output")
        }
        return (flattenAsFloat(logits), flattenAsFloat(softOut))
    }

    // MARK: - Helpers

    private static func loadMainFunction(at url: URL) async throws -> InferenceFunction {
        let model = try await PreparedModel.prepare(at: url).model
        let name = model.functionNames.contains("main") ? "main" : (model.functionNames.first ?? "main")
        guard let fn = try model.loadFunction(named: name) else {
            throw DiffusionRunnerError.contract("cannot load function '\(name)' from \(url.lastPathComponent)")
        }
        return fn
    }

    private static func decoderVocabSize(_ desc: InferenceFunctionDescriptor) throws -> Int {
        guard case .ndArray(let logitsDesc) = desc.outputDescriptor(of: "logits") else {
            throw DiffusionRunnerError.contract("decoder missing 'logits' output descriptor")
        }
        return logitsDesc.shape.last ?? 0
    }

    private static func argmaxRow(_ logits: [Float], row: Int, vocab: Int) -> Int32 {
        let base = row * vocab
        var best = 0
        var bestVal = logits[base]
        for j in 1..<vocab where logits[base + j] > bestVal {
            bestVal = logits[base + j]
            best = j
        }
        return Int32(best)
    }

    private static func entropyRow(_ logits: [Float], row: Int, vocab: Int) -> Float {
        let base = row * vocab
        var maxV = logits[base]
        for j in 1..<vocab where logits[base + j] > maxV { maxV = logits[base + j] }
        var sum: Float = 0
        for j in 0..<vocab { sum += expf(logits[base + j] - maxV) }
        let logSum = logf(sum) + maxV
        var ent: Float = 0
        for j in 0..<vocab {
            let p = expf(logits[base + j] - logSum)
            if p > 0 { ent -= p * (logits[base + j] - logSum) }
        }
        return ent
    }

    private static func sampleRow(_ logits: [Float], row: Int, vocab: Int) -> Int32 {
        let base = row * vocab
        var maxV = logits[base]
        for j in 1..<vocab where logits[base + j] > maxV { maxV = logits[base + j] }
        var sum: Float = 0
        for j in 0..<vocab { sum += expf(logits[base + j] - maxV) }
        let r = Float.random(in: 0..<1) * sum
        var acc: Float = 0
        for j in 0..<vocab {
            acc += expf(logits[base + j] - maxV)
            if acc >= r { return Int32(j) }
        }
        return Int32(vocab - 1)
    }
}

enum DiffusionRunnerError: Error, CustomStringConvertible {
    case contract(String)
    var description: String {
        switch self {
        case .contract(let m): return "DiffusionGemma runner: \(m)"
        }
    }
}
