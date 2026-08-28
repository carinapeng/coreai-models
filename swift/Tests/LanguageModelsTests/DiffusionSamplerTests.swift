// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("DiffusionSampler")
struct DiffusionSamplerTests {
    @Test func temperatureSchedule() {
        // step == maxSteps -> t_max; the final small step approaches t_min.
        #expect(DiffusionSampler.temperature(step: 16, maxSteps: 16, tMin: 0.4, tMax: 0.8) == 0.8)
        let last = DiffusionSampler.temperature(step: 1, maxSteps: 16, tMin: 0.4, tMax: 0.8)
        #expect(abs(last - (0.4 + 0.4 / 16.0)) < 1e-5)
    }

    @Test func argmaxPicksMaxPerRow() {
        // 2 rows, vocab 4.
        let logits: [Float] = [0.1, 0.2, 5.0, 0.3, 9.0, 0.0, 0.0, 0.0]
        #expect(DiffusionSampler.argmaxRow(logits, row: 0, vocab: 4) == 2)
        #expect(DiffusionSampler.argmaxRow(logits, row: 1, vocab: 4) == 0)
    }

    @Test func entropyOneHotIsZeroUniformIsLogVocab() {
        let vocab = 8
        var onehot = [Float](repeating: -1e4, count: vocab)
        onehot[0] = 1e4
        let uniform = [Float](repeating: 0, count: vocab)
        #expect(DiffusionSampler.entropyRow(onehot, row: 0, vocab: vocab) < 1e-2)
        let uni = DiffusionSampler.entropyRow(uniform, row: 0, vocab: vocab)
        #expect(abs(uni - Float(log(Double(vocab)))) < 1e-3)
    }

    @Test func sampleIsDeterministicOnOneHot() {
        let vocab = 5
        var onehot = [Float](repeating: -1e4, count: vocab)
        onehot[3] = 1e4
        // A near-one-hot distribution samples the peak for any interior uniform draw.
        #expect(DiffusionSampler.sampleRow(onehot, row: 0, vocab: vocab, u: 0.01) == 3)
        #expect(DiffusionSampler.sampleRow(onehot, row: 0, vocab: vocab, u: 0.5) == 3)
        #expect(DiffusionSampler.sampleRow(onehot, row: 0, vocab: vocab, u: 0.99) == 3)
    }

    @Test func acceptMaskRespectsEntropyBound() {
        // Sorted ascending: idx1(0.05), idx2(0.2), idx0(0.5). Cumulative-excluding-self
        // at each: 0, 0.05, 0.25.
        let ent: [Float] = [0.5, 0.05, 0.2]
        #expect(DiffusionSampler.acceptMask(entropies: ent, bound: 0.1) == [false, true, true])
        // bound 0 accepts only the single lowest-entropy position (cum 0 <= 0).
        #expect(DiffusionSampler.acceptMask(entropies: ent, bound: 0.0) == [false, true, false])
    }
}
