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
/// Loads the `encoder.aimodel` and `decoder.aimodel` components by path — the
/// `diffusion_llm` bundle schema (encoder/decoder components) is not modeled by
/// the standard `LanguageBundle` loader — then runs the reference block-diffusion
/// algorithm: the encoder prefills the KV cache, and the bidirectional decoder
/// denoises a fixed-length canvas via random init, entropy-bound accept/renoise,
/// self-conditioning, and a linear temperature schedule, stopping when the
/// argmax canvas is stable and confident.
///
/// All tensor shapes (layers, KV heads, head_dim, hidden size, canvas length,
/// vocab) are read from the exported function descriptors; only the diffusion
/// hyperparameters and the chat format are constants. The encoder and decoder
/// must be static-shape exports (see `export_diffusion_gemma.py --static-encoder`):
/// the encoder is exported for a fixed prompt length, so the prompt must
/// tokenize to exactly that many tokens.
enum DiffusionGemmaRunner {
    // Diffusion hyperparameters (defaults from the checkpoint generation_config).
    static let tMax: Float = 0.8
    static let tMin: Float = 0.4
    static let entropyBound: Float = 0.1
    static let confidenceThreshold: Float = 0.005
    // DiffusionGemma chat format and beginning-of-sequence token.
    static let bosToken = 2
    static let eosToken: Int32 = 1
    static let endOfTurnToken: Int32 = 106

    static func chatFormat(_ prompt: String) -> String {
        "<|turn>user\n\(prompt)<turn|>\n<|turn>model\n"
    }

    static func run(
        bundleDir: String,
        prompt: String,
        maxSteps: Int,
        verbose: Bool
    ) async throws {
        func log(_ message: @autoclosure () -> String) {
            if verbose { print(message()) }
        }

        let dir = URL(fileURLWithPath: bundleDir, isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: dir.appendingPathComponent("tokenizer"))
        let ids = [bosToken] + tokenizer.encode(text: chatFormat(prompt), addSpecialTokens: false)
        let encLen = ids.count

        let encFn = try await loadMainFunction(at: dir.appendingPathComponent("encoder.aimodel"))
        let decFn = try await loadMainFunction(at: dir.appendingPathComponent("decoder.aimodel"))
        log("Loaded encoder/decoder; prompt tokenizes to \(encLen) tokens")

        // ---- Cache/model geometry, read from the descriptors. ----------------
        // Encoder keyCache input is [layers, 1, kvHeads, seq, headDim].
        let encDesc = encFn.descriptor
        let cacheShape = try inputShape(encDesc, "keyCache", label: "encoder")
        let cacheSeq = cacheShape[3]
        guard cacheSeq == encLen else {
            throw DiffusionRunnerError.contract(
                "this bundle's encoder is exported for a \(cacheSeq)-token prompt, but the prompt "
                    + "tokenizes to \(encLen). Re-export with --enc-len \(encLen), or use a "
                    + "prompt of \(cacheSeq) tokens.")
        }
        // Decoder decoder_input_ids is [1, canvas]; prev_soft_embeds is [1, canvas, hidden].
        let decDesc = decFn.descriptor
        let canvasLength = try inputShape(decDesc, "decoder_input_ids", label: "decoder")[1]
        let hiddenSize = try inputShape(decDesc, "prev_soft_embeds", label: "decoder")[2]
        let vocab = try outputShape(decDesc, "logits", label: "decoder").last ?? 0

        // ---- 1. Encoder prefill (functional cache: zeros in, filled out). ----
        // The static encoder is functional: keyCache/valueCache are INPUTS (zeros
        // in) and OUTPUTS (filled out), not persistent state.
        let cacheCount = cacheShape.reduce(1, *)
        let cacheZeros = [Float](repeating: 0, count: cacheCount)

        var encInIds = NDArray(descriptor: try inDesc(encDesc, "input_ids", label: "encoder"))
        fillNDArray(&encInIds, as: Int32.self, count: encLen) { Int32(ids[$0]) }
        var encPos = NDArray(descriptor: try inDesc(encDesc, "position_ids", label: "encoder"))
        fillNDArray(&encPos, as: Int32.self, count: encLen) { Int32($0) }
        var keyIn = NDArray(descriptor: try inDesc(encDesc, "keyCache", label: "encoder"))
        fillFloatNDArray(&keyIn, with: cacheZeros)
        var valueIn = NDArray(descriptor: try inDesc(encDesc, "valueCache", label: "encoder"))
        fillFloatNDArray(&valueIn, with: cacheZeros)

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
        let encoderK = flattenAsFloat(keyOut)
        let encoderV = flattenAsFloat(valueOut)
        log("Encoder prefill complete")

        // ---- 2. Denoising loop over one canvas. ------------------------------
        var canvas = (0..<canvasLength).map { _ in Int32.random(in: 0..<Int32(vocab)) }
        var soft = [Float](repeating: 0, count: canvasLength * hiddenSize)
        var prevArgmax: [Int32]? = nil
        var bestArgmax = canvas
        var bestMeanH = Float.greatestFiniteMagnitude

        for step in stride(from: maxSteps, through: 1, by: -1) {
            let temp = tMin + (tMax - tMin) * (Float(step) / Float(maxSteps))
            let (processed, softOut) = try await runDecoder(
                decFn, decDesc, canvas: canvas, soft: soft, encLen: encLen,
                canvasLength: canvasLength, hiddenSize: hiddenSize,
                encoderK: encoderK, encoderV: encoderV, temperature: temp)
            soft = softOut

            let argmaxCanvas = (0..<canvasLength).map { argmaxRow(processed, row: $0, vocab: vocab) }
            let entropies = (0..<canvasLength).map { entropyRow(processed, row: $0, vocab: vocab) }
            let (accepted, next) = acceptAndRenoise(processed, entropies: entropies, current: canvas, vocab: vocab)
            canvas = next

            let meanH = entropies.reduce(0, +) / Float(canvasLength)
            if meanH < bestMeanH {
                bestMeanH = meanH
                bestArgmax = argmaxCanvas
            }
            if verbose {
                let preview = tokenizer.decode(tokens: argmaxCanvas.map(Int.init).filter { $0 != 0 })
                log(
                    "  step \(step) temp=\(fmt(temp)) meanH=\(fmt(meanH)) "
                        + "accept=\(accepted)/\(canvasLength) | "
                        + preview.replacingOccurrences(of: "\n", with: " ").prefix(80))
            }

            let stable = prevArgmax.map { $0 == argmaxCanvas } ?? false
            prevArgmax = argmaxCanvas
            if stable && meanH < confidenceThreshold { break }
        }

        // Return the most-confident step's canvas, trimmed at end-of-turn / eos.
        var finalIds: [Int] = []
        for t in bestArgmax {
            if t == eosToken || t == endOfTurnToken { break }
            if t != 0 { finalIds.append(Int(t)) }
        }
        print(tokenizer.decode(tokens: finalIds))
    }

    // MARK: - Decoder step

    private static func runDecoder(
        _ fn: InferenceFunction,
        _ desc: InferenceFunctionDescriptor,
        canvas: [Int32],
        soft: [Float],
        encLen: Int,
        canvasLength: Int,
        hiddenSize: Int,
        encoderK: [Float],
        encoderV: [Float],
        temperature: Float
    ) async throws -> (processed: [Float], soft: [Float]) {
        var dIds = NDArray(descriptor: try inDesc(desc, "decoder_input_ids", label: "decoder"))
        fillNDArray(&dIds, as: Int32.self, count: canvasLength) { canvas[$0] }
        var prevSoft = NDArray(descriptor: try inDesc(desc, "prev_soft_embeds", label: "decoder"))
        fillFloatNDArray(&prevSoft, with: soft)
        var pos = NDArray(descriptor: try inDesc(desc, "position_ids", label: "decoder"))
        fillNDArray(&pos, as: Int32.self, count: canvasLength) { Int32(encLen + $0) }
        var ek = NDArray(descriptor: try inDesc(desc, "encoder_k", label: "decoder"))
        fillFloatNDArray(&ek, with: encoderK)
        var ev = NDArray(descriptor: try inDesc(desc, "encoder_v", label: "decoder"))
        fillFloatNDArray(&ev, with: encoderV)
        var tempArr = NDArray(descriptor: try inDesc(desc, "temperature", label: "decoder"))
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

    // MARK: - Sampler

    /// Entropy-bound acceptance + renoising. Accepts the lowest-entropy positions
    /// whose cumulative (excluding self) entropy stays within `entropyBound`,
    /// samples those from the denoiser, and re-randomizes the rest.
    private static func acceptAndRenoise(
        _ logits: [Float], entropies: [Float], current: [Int32], vocab: Int
    ) -> (accepted: Int, next: [Int32]) {
        let n = current.count
        let order = (0..<n).sorted { entropies[$0] < entropies[$1] }
        var accept = [Bool](repeating: false, count: n)
        var cum: Float = 0
        for idx in order {
            if cum <= entropyBound { accept[idx] = true }
            cum += entropies[idx]
        }
        var next = [Int32](repeating: 0, count: n)
        for i in 0..<n {
            next[i] = accept[i] ? sampleRow(logits, row: i, vocab: vocab) : Int32.random(in: 0..<Int32(vocab))
        }
        return (accept.filter { $0 }.count, next)
    }

    // MARK: - Descriptor / math helpers

    private static func loadMainFunction(at url: URL) async throws -> InferenceFunction {
        let model = try await PreparedModel.prepare(at: url).model
        let name = model.functionNames.contains("main") ? "main" : (model.functionNames.first ?? "main")
        guard let fn = try model.loadFunction(named: name) else {
            throw DiffusionRunnerError.contract("cannot load function '\(name)' from \(url.lastPathComponent)")
        }
        return fn
    }

    private static func inDesc(
        _ desc: InferenceFunctionDescriptor, _ name: String, label: String
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let d) = desc.inputDescriptor(of: name) else {
            throw DiffusionRunnerError.contract("\(label) missing input '\(name)'")
        }
        return d
    }

    private static func inputShape(
        _ desc: InferenceFunctionDescriptor, _ name: String, label: String
    ) throws -> [Int] {
        try inDesc(desc, name, label: label).shape
    }

    private static func outputShape(
        _ desc: InferenceFunctionDescriptor, _ name: String, label: String
    ) throws -> [Int] {
        guard case .ndArray(let d) = desc.outputDescriptor(of: name) else {
            throw DiffusionRunnerError.contract("\(label) missing output '\(name)'")
        }
        return d.shape
    }

    private static func fmt(_ x: Float) -> String { String(format: "%.3f", x) }

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
