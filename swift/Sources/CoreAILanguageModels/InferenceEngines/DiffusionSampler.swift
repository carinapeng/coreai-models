// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Pure block-diffusion sampler math, independent of the Core AI runtime.
///
/// Factored out of the DiffusionGemma runner so it can be unit-tested and shared
/// with a future `DiffusionEngine`. All functions operate on a flat row-major
/// logits buffer of shape `[canvas, vocab]` (row `r` starts at `r * vocab`).
public enum DiffusionSampler {
    /// Linear temperature schedule: `t_min + (t_max - t_min) * (step / maxSteps)`.
    /// Steps count down from `maxSteps` to 1, so early (high-`step`) iterations
    /// are hotter.
    public static func temperature(step: Int, maxSteps: Int, tMin: Float, tMax: Float) -> Float {
        tMin + (tMax - tMin) * (Float(step) / Float(maxSteps))
    }

    /// Index of the maximum logit in row `row`.
    public static func argmaxRow(_ logits: [Float], row: Int, vocab: Int) -> Int32 {
        let base = row * vocab
        var best = 0
        var bestVal = logits[base]
        for j in 1..<vocab where logits[base + j] > bestVal {
            bestVal = logits[base + j]
            best = j
        }
        return Int32(best)
    }

    /// Shannon entropy (nats) of `softmax(row)`.
    public static func entropyRow(_ logits: [Float], row: Int, vocab: Int) -> Float {
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

    /// Inverse-CDF sample from `softmax(row)` given a uniform draw `u` in `[0, 1)`.
    /// Passing `u` explicitly keeps this deterministic and testable; callers pass
    /// `Float.random(in: 0..<1)`.
    public static func sampleRow(_ logits: [Float], row: Int, vocab: Int, u: Float) -> Int32 {
        let base = row * vocab
        var maxV = logits[base]
        for j in 1..<vocab where logits[base + j] > maxV { maxV = logits[base + j] }
        var sum: Float = 0
        for j in 0..<vocab { sum += expf(logits[base + j] - maxV) }
        let target = u * sum
        var acc: Float = 0
        for j in 0..<vocab {
            acc += expf(logits[base + j] - maxV)
            if acc >= target { return Int32(j) }
        }
        return Int32(vocab - 1)
    }

    /// Entropy-bound acceptance: accept the lowest-entropy positions whose
    /// cumulative (excluding self) entropy stays within `bound`.
    public static func acceptMask(entropies: [Float], bound: Float) -> [Bool] {
        let n = entropies.count
        let order = (0..<n).sorted { entropies[$0] < entropies[$1] }
        var accept = [Bool](repeating: false, count: n)
        var cum: Float = 0
        for idx in order {
            if cum <= bound { accept[idx] = true }
            cum += entropies[idx]
        }
        return accept
    }
}
