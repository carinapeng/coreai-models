// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Selection rules of the greedy row scan: the lowest index wins a tie, a NaN ranks as negative
/// infinity, and an empty row selects index 0.
@Suite("Logit row scan")
struct LogitRowScanTests {
    private func argmax(_ row: [Float]) -> Int {
        row.withUnsafeBufferPointer { LogitRowScan.argmax($0) }
    }

    @Test("The first maximum wins a tie, whatever the row length")
    func firstMaximumWinsTies() {
        var rng = Xoshiro256StarStar(seed: 21)
        for n in [1, 2, 7, 8, 9, 31, 64, 1000, 4097, 262_144] {
            for _ in 0..<5 {
                let row = (0..<n).map { _ in Float(rng.next() % 3) }
                let expected = row.indices.first { row[$0] == row.max()! }!
                #expect(argmax(row) == expected, "n \(n)")
            }
        }
    }

    @Test("An empty row selects index 0")
    func emptyRow() {
        #expect(argmax([]) == 0)
    }

    @Test("A NaN never beats a number, including at index 0")
    func nanRanksLowest() {
        #expect(argmax([.nan, 0, 5, 1]) == 2)
        #expect(argmax([2, .nan, 1]) == 0)
        #expect(argmax([-3, .nan, -1, -1]) == 2)
    }

    @Test("A NaN ties with negative infinity, and the lowest index wins")
    func nanTiesWithNegativeInfinity() {
        #expect(argmax([.nan, .nan]) == 0)
        #expect(argmax([.nan, -.infinity]) == 0)
        #expect(argmax([-.infinity, .nan]) == 0)
        #expect(argmax([-.infinity, -.infinity, -2]) == 2)
    }

    @Test("Opposing infinities take the exact path and still pick the maximum")
    func opposingInfinities() {
        #expect(argmax([-.infinity, 1, .infinity, 3]) == 2)
        #expect(argmax([.infinity, .infinity, -.infinity]) == 0)
    }
}
