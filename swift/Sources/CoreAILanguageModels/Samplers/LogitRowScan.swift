// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Greedy selection over one logit row widened to `Float`.
///
/// - Ties: the lowest index wins, as with `vDSP_maxvi`.
/// - NaN: a NaN entry ranks as negative infinity. It never beats a number, and ties with other
///   negative-infinity entries break toward the lowest index. An empty row selects index 0.
///   `vDSP_maxvi` returns an arbitrary index when the row holds a NaN, so rows are checked first.
enum LogitRowScan {
    /// Index of the first maximum of `row` under the NaN rule above.
    static func argmax(_ row: UnsafeBufferPointer<Float>) -> Int {
        guard let base = row.baseAddress, !row.isEmpty else { return 0 }
        if hasNaNOrOpposingInfinities(row) { return scalarArgmax(row) }
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(base, 1, &maxValue, &maxIndex, vDSP_Length(row.count))
        return Int(maxIndex)
    }

    /// Whether `row` may hold a NaN. The sum of a row is NaN when it does, and also when it holds
    /// both infinities; callers take the exact scalar path in either case.
    static func hasNaNOrOpposingInfinities(_ row: UnsafeBufferPointer<Float>) -> Bool {
        guard let base = row.baseAddress, !row.isEmpty else { return false }
        var sum: Float = 0
        vDSP_sve(base, 1, &sum, vDSP_Length(row.count))
        return sum.isNaN
    }

    private static func scalarArgmax(_ row: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        var bestValue = -Float.infinity
        for i in 0..<row.count where row[i] > bestValue {
            bestValue = row[i]
            best = i
        }
        return best
    }
}
