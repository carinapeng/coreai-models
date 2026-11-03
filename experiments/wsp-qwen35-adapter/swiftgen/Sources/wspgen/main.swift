// wspgen: generation and timing driver for the Qwen3.5 Core AI export with a rank-1
// mutable-weight adapter (states l15_u, l15_c, l19_u, l19_c next to the model's caches).
//
// Loads like the repo's Swift LLM runner does for dynamic-shape graphs: GPU preferred,
// expectFrequentReshapes. The prompt runs on `prefill`, decode on `main` one token at a time.
// The host owns every state; the adapter is set by filling its state arrays between calls.
//
// usage:
//   wspgen gen   <model.aimodel> <adapter.bin> <jobs.jsonl> <out.jsonl> <stop ids comma-sep> [max ctx]
//   wspgen bench <model.aimodel> <adapter.bin> <out.json>
//
// adapter.bin: float32 little-endian l15_u(1024) l15_c(3584) l19_u(1024) l19_c(3584).
// jobs.jsonl rows: {"key", "prompt": [ids], "scale", "seed", "cap", "temperature", "top_p",
//                   "top_k", "presence_penalty"}
// out.jsonl rows:  {"key", "tokens", "finish", "prefill_seconds", "decode_seconds"}

import CoreAI
import Darwin
import Foundation

struct RuntimeError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

@inline(__always)
@_unsafeNonescapableResult
@_lifetime(immortal)
func escapeViews(_ views: consuming InferenceFunction.MutableViews) -> InferenceFunction.MutableViews {
    views
}

func elementCount(_ a: NDArray) -> Int { a.shape.reduce(1, *) }

func zeroFill(_ array: inout NDArray) {
    let width: Int
    switch array.scalarType {
    case .float16: width = 2
    case .float32, .int32: width = 4
    default: width = 4
    }
    let n = elementCount(array) * width
    array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in memset(ptr, 0, n) }
}

/// Stride-aware fp16 fill of the first `values.count` logical elements.
func fillF16(_ array: inout NDArray, _ values: [Float]) {
    let view = array.mutableRawView().view(as: Float16.self)
    view.withUnsafeMutablePointer { ptr, shape, strides in
        let rank = shape.count
        var idx = [Int](repeating: 0, count: rank)
        for i in 0..<values.count {
            var off = 0
            for d in 0..<rank { off += idx[d] * strides[d] }
            ptr[off] = Float16(values[i])
            var d = rank - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < shape[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
    }
}

func fillI32(_ array: inout NDArray, count: Int, _ gen: (Int) -> Int32) {
    let view = array.mutableRawView().view(as: Int32.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<count { ptr[i] = gen(i) }  // (1, n) int32: contiguous
    }
}

func readF16(_ array: NDArray, count: Int, into out: inout [Float]) {
    array.view(as: Float16.self).withUnsafePointer { ptr, shape, strides in
        // logits (1, 1, V): last dim contiguous
        precondition(strides[strides.count - 1] == 1)
        for i in 0..<count { out[i] = Float(ptr[i]) }
    }
}

final class Session {
    let fnMain: InferenceFunction
    let fnPrefill: InferenceFunction
    let descMain: InferenceFunctionDescriptor
    let descPrefill: InferenceFunctionDescriptor
    var arrays: [String: NDArray] = [:]
    let stateNames: [String]
    let adapterNames = ["l15_u", "l15_c", "l19_u", "l19_c"]
    let hasAdapter: Bool
    let vocab: Int
    var pos = 0
    var logitsArray: NDArray
    var logits: [Float]

    init(model: AIModel, maxCtx: Int) throws {
        guard let m = try model.loadFunction(named: "main"), let p = try model.loadFunction(named: "prefill"),
            let dm = model.functionDescriptor(for: "main"), let dp = model.functionDescriptor(for: "prefill")
        else { throw RuntimeError("missing main/prefill") }
        fnMain = m
        fnPrefill = p
        descMain = dm
        descPrefill = dp
        stateNames = dm.stateNames
        hasAdapter = stateNames.contains("l15_u")
        for name in stateNames {
            guard case .ndArray(let d) = dm.stateDescriptor(of: name) else { throw RuntimeError("state \(name)") }
            let shape = d.shape.map { $0 < 0 ? maxCtx : $0 }
            var a = NDArray(descriptor: d.resolvingDynamicDimensions(shape))
            zeroFill(&a)
            arrays[name] = a
        }
        guard case .ndArray(let ld) = dm.outputDescriptor(of: "logits") else { throw RuntimeError("logits") }
        vocab = ld.shape.last!
        logitsArray = NDArray(descriptor: ld.resolvingDynamicDimensions([1, 1, vocab]))
        logits = [Float](repeating: 0, count: vocab)
    }

    func resetCaches() {
        for name in stateNames where !adapterNames.contains(name) { zeroFill(&arrays[name]!) }
        pos = 0
    }

    func setAdapter(_ vectors: [[Float]], scale: Float) {
        guard hasAdapter else { return }
        for (k, name) in adapterNames.enumerated() {
            let v = name.hasSuffix("_u") ? vectors[k].map { $0 * scale } : vectors[k]
            fillF16(&arrays[name]!, v)
        }
    }

    @_lifetime(views: borrow self)
    func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    func inputs(_ desc: InferenceFunctionDescriptor, _ tokens: [Int32]) throws -> [String: NDArray] {
        guard case .ndArray(let idsD) = desc.inputDescriptor(of: "input_ids"),
            case .ndArray(let posD) = desc.inputDescriptor(of: "position_ids")
        else { throw RuntimeError("inputs") }
        var ids = NDArray(descriptor: idsD.resolvingDynamicDimensions([1, tokens.count]))
        fillI32(&ids, count: tokens.count) { tokens[$0] }
        let total = pos + tokens.count
        var p = NDArray(descriptor: posD.resolvingDynamicDimensions([1, total]))
        fillI32(&p, count: total) { Int32($0) }
        return ["input_ids": ids, "position_ids": p]
    }

    func prefill(_ tokens: [Int32]) async throws {
        let inp = try inputs(descPrefill, tokens)
        var states = InferenceFunction.MutableViews()
        bind(into: &states)
        _ = try await fnPrefill.run(
            inputs: inp, states: escapeViews(consume states),
            outputViews: InferenceFunction.MutableViews())
        pos += tokens.count
    }

    /// One decode step on `main`; logits land in `self.logits`.
    func step(_ token: Int32) async throws {
        let inp = try inputs(descMain, [token])
        var states = InferenceFunction.MutableViews()
        bind(into: &states)
        var out = InferenceFunction.MutableViews()
        out.insert(&logitsArray, for: "logits")
        _ = try await fnMain.run(inputs: inp, states: escapeViews(consume states), outputViews: consume out)
        pos += 1
        readF16(logitsArray, count: vocab, into: &logits)
    }
}

struct SplitMix64 {
    var s: UInt64
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

/// presence penalty -> temperature -> top-k -> top-p -> sample.
func sample(
    _ logits: [Float], seen: [Bool], temperature: Float, topK: Int, topP: Float,
    presence: Float, rng: inout SplitMix64
) -> Int {
    var topIdx = [Int]()
    var topVal = [Float]()
    topIdx.reserveCapacity(topK + 1)
    topVal.reserveCapacity(topK + 1)
    var minVal = -Float.infinity
    for i in 0..<logits.count {
        var x = logits[i]
        if seen[i] { x -= presence }
        if topVal.count == topK && x <= minVal { continue }
        var j = topVal.count
        while j > 0 && topVal[j - 1] < x { j -= 1 }
        topVal.insert(x, at: j)
        topIdx.insert(i, at: j)
        if topVal.count > topK {
            topVal.removeLast()
            topIdx.removeLast()
        }
        if topVal.count == topK { minVal = topVal[topK - 1] }
    }
    let mx = topVal[0]
    var p = topVal.map { Double(exp(($0 - mx) / temperature)) }
    let sum = p.reduce(0, +)
    p = p.map { $0 / sum }
    var keep = 0
    var cum = 0.0
    while keep < p.count {  // smallest prefix with mass >= top_p
        cum += p[keep]
        keep += 1
        if cum >= Double(topP) { break }
    }
    let mass = p[0..<keep].reduce(0, +)
    var u = rng.uniform() * mass
    for k in 0..<keep {
        u -= p[k]
        if u <= 0 { return topIdx[k] }
    }
    return topIdx[keep - 1]
}

func loadAdapter(_ path: String) throws -> [[Float]] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let all: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let sizes = [1024, 3584, 1024, 3584]
    var out: [[Float]] = []
    var o = 0
    for s in sizes {
        out.append(Array(all[o..<(o + s)]))
        o += s
    }
    precondition(o == all.count, "adapter.bin size")
    return out
}

func loadModel(_ path: String) async throws -> AIModel {
    var opts = SpecializationOptions(preferredComputeUnitKind: .gpu)
    opts.expectFrequentReshapes = true
    return try await AIModel(contentsOf: URL(fileURLWithPath: path), options: opts)
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

func runGen(_ args: [String]) async throws {
    let (modelPath, adapterPath, jobsPath, outPath) = (args[0], args[1], args[2], args[3])
    let stops = Set(args[4].split(separator: ",").map { Int($0)! })
    let maxCtx = args.count > 5 ? Int(args[5])! : 9216
    var done = Set<String>()
    if let text = try? String(contentsOfFile: outPath, encoding: .utf8) {
        for line in text.split(separator: "\n") {
            if let o = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                let k = o["key"] as? String
            {
                done.insert(k)
            }
        }
    }
    let jobs = try String(contentsOfFile: jobsPath, encoding: .utf8).split(separator: "\n").compactMap {
        try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
    }
    let todo = jobs.filter { !done.contains($0["key"] as! String) }
    print("jobs \(jobs.count), to do \(todo.count)")
    fflush(stdout)
    let t0 = now()
    let model = try await loadModel(modelPath)
    let s = try Session(model: model, maxCtx: maxCtx)
    print("loaded in \(String(format: "%.1f", now() - t0))s, adapter states: \(s.hasAdapter)")
    fflush(stdout)
    let vec = try loadAdapter(adapterPath)
    if !FileManager.default.fileExists(atPath: outPath) {
        FileManager.default.createFile(atPath: outPath, contents: nil)
    }
    let fh = try FileHandle(forWritingTo: URL(fileURLWithPath: outPath))
    try fh.seekToEnd()
    var totalTok = 0
    let tAll = now()
    for job in todo {
        let key = job["key"] as! String
        let prompt = (job["prompt"] as! [Int]).map { Int32($0) }
        let scale = Float(job["scale"] as! Double)
        var rng = SplitMix64(s: UInt64(job["seed"] as! Int))
        let cap = min(job["cap"] as! Int, maxCtx - prompt.count - 1)
        let temperature = Float(job["temperature"] as! Double)
        let topP = Float(job["top_p"] as! Double)
        let topK = job["top_k"] as! Int
        let presence = Float(job["presence_penalty"] as! Double)
        s.setAdapter(vec, scale: scale)
        s.resetCaches()
        let tp = now()
        if prompt.count > 1 { try await s.prefill(Array(prompt.dropLast())) }
        try await s.step(prompt.last!)
        let prefillSec = now() - tp
        var seen = [Bool](repeating: false, count: s.vocab)
        var gen: [Int] = []
        var finish = "length"
        let td = now()
        for i in 0..<cap {
            let t = sample(
                s.logits, seen: seen, temperature: temperature, topK: topK, topP: topP,
                presence: presence, rng: &rng)
            gen.append(t)
            seen[t] = true
            if stops.contains(t) {
                finish = "stop"
                break
            }
            if i < cap - 1 { try await s.step(Int32(t)) }
        }
        let decodeSec = now() - td
        totalTok += gen.count
        let rec: [String: Any] = [
            "key": key, "tokens": gen, "finish": finish,
            "prefill_seconds": prefillSec, "decode_seconds": decodeSec,
        ]
        var line = try JSONSerialization.data(withJSONObject: rec)
        line.append(0x0A)
        try fh.write(contentsOf: line)
        try fh.synchronize()
        print(
            String(
                format: "%@ tokens=%d finish=%@ %.1f tok/s (overall %.1f tok/s)", key, gen.count, finish,
                Double(gen.count) / decodeSec, Double(totalTok) / (now() - tAll)))
        fflush(stdout)
    }
    try fh.close()
}

/// Teacher-forced logits: for each job, prefill token 0, then `main` one token at a time;
/// appends fp16 logits for positions 1..T-1 to <out.bin> (job order).
func runLogits(_ args: [String]) async throws {
    let (modelPath, adapterPath, jobsPath, outPath) = (args[0], args[1], args[2], args[3])
    let model = try await loadModel(modelPath)
    let s = try Session(model: model, maxCtx: 9216)
    let vec = try loadAdapter(adapterPath)
    let jobs = try String(contentsOfFile: jobsPath, encoding: .utf8).split(separator: "\n").compactMap {
        try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
    }
    FileManager.default.createFile(atPath: outPath, contents: nil)
    let fh = try FileHandle(forWritingTo: URL(fileURLWithPath: outPath))
    for job in jobs {
        let prompt = (job["prompt"] as! [Int]).map { Int32($0) }
        s.setAdapter(vec, scale: Float(job["scale"] as! Double))
        s.resetCaches()
        try await s.prefill([prompt[0]])
        for t in prompt.dropFirst() {
            try await s.step(t)
            let half = s.logits.map { Float16($0) }
            try fh.write(contentsOf: half.withUnsafeBufferPointer { Data(buffer: $0) })
        }
        print("\(job["key"] as! String): \(prompt.count - 1) positions")
        fflush(stdout)
    }
    try fh.close()
}

func runBench(_ args: [String]) async throws {
    let (modelPath, adapterPath, outPath) = (args[0], args[1], args[2])
    let prefillLen = 512
    let decodeSteps = 256
    let reps = 3
    let t0 = now()
    let model = try await loadModel(modelPath)
    let s = try Session(model: model, maxCtx: 9216)
    let loadSec = now() - t0
    let vec = try loadAdapter(adapterPath)
    var rng = SplitMix64(s: 0)
    let prompt = (0..<prefillLen).map { _ in Int32(1000 + Int(rng.next() % 99000)) }
    var result: [String: Any] = [
        "load_seconds": loadSec, "prefill_len": prefillLen,
        "decode_steps": decodeSteps, "reps": reps, "adapter_states": s.hasAdapter,
    ]
    func one(_ label: String, scale: Float) async throws {
        var pre: [Double] = []
        var dec: [Double] = []
        for r in 0..<(reps + 1) {  // first rep is warm-up
            s.setAdapter(vec, scale: scale)
            s.resetCaches()
            var t = now()
            try await s.prefill(prompt)
            let p = now() - t
            var tok: Int32 = 1234
            t = now()
            for _ in 0..<decodeSteps {
                try await s.step(tok)
                var best = 0
                for i in 1..<s.vocab where s.logits[i] > s.logits[best] { best = i }
                tok = Int32(best)
            }
            let d = now() - t
            if r > 0 {
                pre.append(p)
                dec.append(d)
            }
        }
        pre.sort()
        dec.sort()
        result[label] = [
            "prefill_tok_per_s": Double(prefillLen) / pre[reps / 2],
            "decode_tok_per_s": Double(decodeSteps) / dec[reps / 2],
            "prefill_seconds": pre, "decode_seconds": dec,
        ]
        print(
            String(
                format: "%@: prefill %.1f tok/s, decode %.1f tok/s", label,
                Double(prefillLen) / pre[reps / 2], Double(decodeSteps) / dec[reps / 2]))
        fflush(stdout)
    }
    if s.hasAdapter {
        try await one("adapter_off", scale: 0)
        try await one("adapter_on", scale: 1)
        // Toggle cost: host fill of all four states, and the decode step right after a toggle.
        s.resetCaches()
        try await s.prefill(Array(prompt.prefix(64)))
        var fills: [Double] = []
        var after: [Double] = []
        var steady: [Double] = []
        var tok: Int32 = 1234
        for i in 0..<200 {
            if i % 10 == 0 {
                let t = now()
                s.setAdapter(vec, scale: (i / 10) % 2 == 0 ? 1 : 0)
                fills.append(now() - t)
                let t2 = now()
                try await s.step(tok)
                after.append(now() - t2)
            } else {
                let t = now()
                try await s.step(tok)
                steady.append(now() - t)
            }
            tok = Int32(1000 + i)
        }
        func med(_ x: [Double]) -> Double {
            let y = x.sorted()
            return y[y.count / 2]
        }
        result["toggle"] = [
            "n_toggles": fills.count, "set_adapter_ms_median": 1e3 * med(fills),
            "set_adapter_ms_max": 1e3 * fills.max()!,
            "step_after_toggle_ms_median": 1e3 * med(after),
            "steady_step_ms_median": 1e3 * med(steady),
        ]
        print("toggle:", result["toggle"]!)
    } else {
        try await one("plain", scale: 0)
    }
    let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted])
    try data.write(to: URL(fileURLWithPath: outPath))
}

let argv = CommandLine.arguments
do {
    switch argv.count > 1 ? argv[1] : "" {
    case "gen": try await runGen(Array(argv.dropFirst(2)))
    case "bench": try await runBench(Array(argv.dropFirst(2)))
    case "logits": try await runLogits(Array(argv.dropFirst(2)))
    default:
        print("usage: wspgen gen|bench ...")
        exit(2)
    }
} catch {
    print("error: \(error)")
    exit(1)
}
