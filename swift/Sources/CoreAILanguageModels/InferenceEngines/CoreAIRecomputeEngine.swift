// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Synchronization

// MARK: - Core AI Stateless Full-Recompute Engine

/// Stateless full-recompute engine for fixed-shape static-input exports.
public final class CoreAIRecomputeEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }
    public var vocabSize: Int { config.vocabSize }
    public let config: ModelConfig

    // Core AI function handle
    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    // I/O names from descriptor
    private let inputIdsName: String
    private let positionIdsName: String
    private let cacheInputNames: [String]
    private let cacheStateNames: [String]
    private let logitsName: String

    private let maxLen: Int
    private let dynamicSeq: Bool

    // Descriptors retained for per-call (re)allocation.
    private let inputIdsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor
    private let cacheDescriptors: [String: NDArrayDescriptor]

    // Persistent buffers reused across steps (only when shapes are static).
    private var inputIdsArray: NDArray
    private var positionIdsArray: NDArray
    private var logitsArray: NDArray

    // Track in-flight generation for drain (same pattern as other engines).
    private let generating = Mutex(false)

    // MARK: - Init

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        self.config = config

        let model = preparedModel.model

        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(config.function)' in model")
        }
        self.functionDescriptor = descriptor

        // Identify token/position inputs and treat all remaining inputs as caches.
        let allInputs = descriptor.inputNames
        guard let idsName = allInputs.first(where: { $0.contains("input_ids") }) ?? allInputs.first,
            let posName = allInputs.first(where: { $0.contains("position") })
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Could not find input_ids / position_ids among inputs: \(allInputs)")
        }
        self.inputIdsName = idsName
        self.positionIdsName = posName
        self.cacheInputNames = allInputs.filter { $0 != idsName && $0 != posName }
        self.cacheStateNames = descriptor.stateNames

        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count): \(descriptor.outputNames)")
        }
        self.logitsName = descriptor.outputNames.first { $0.contains("logits") }
            ?? descriptor.outputNames[0]

        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: idsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(idsName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: posName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(posName)'")
        }
        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }

        self.inputIdsDescriptor = inputIdsDesc
        self.positionIdsDescriptor = posIdsDesc
        self.logitsDescriptor = logitsDesc

        var cdescs: [String: NDArrayDescriptor] = [:]
        for name in cacheInputNames {
            guard case .ndArray(let cdesc) = descriptor.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for cache input '\(name)'")
            }
            cdescs[name] = cdesc
        }
        for name in cacheStateNames {
            guard case .ndArray(let cdesc) = descriptor.stateDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for cache state '\(name)'")
            }
            cdescs[name] = cdesc
        }
        self.cacheDescriptors = cdescs

        let inLast = inputIdsDesc.shape.last ?? -1
        if inLast > 0 {
            self.maxLen = inLast
            self.dynamicSeq = false
        } else {
            guard let anyCache = cdescs.values.first else {
                throw InferenceRuntimeError.invalidState(
                    "Dynamic input_ids but no cache descriptor to derive context length")
            }
            let seqDim = anyCache.shape.count >= 2 ? anyCache.shape.count - 2 : anyCache.shape.count - 1
            let cacheSeq = anyCache.shape[seqDim]
            self.maxLen = cacheSeq > 0 ? cacheSeq : config.maxContextLength
            self.dynamicSeq = true
        }

        let seedLen = dynamicSeq ? 1 : maxLen
        self.inputIdsArray = NDArray(descriptor: inputIdsDesc.resolvingDynamicDimensions([1, seedLen]))
        self.positionIdsArray = NDArray(descriptor: posIdsDesc.resolvingDynamicDimensions([1, seedLen]))
        self.logitsArray = NDArray(
            descriptor: logitsDesc.resolvingDynamicDimensions([1, seedLen, config.vocabSize]))

        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError("Cannot load function '\(config.function)'")
        }
        self.function = fn

        CLILogger.log(
            "CoreAI recompute engine initialized — maxLen=\(maxLen), dynamicSeq=\(dynamicSeq), "
                + "ids='\(idsName)', pos='\(posName)', cacheInputs=\(cacheInputNames), "
                + "cacheStates=\(cacheStateNames), logits='\(logitsName)' \(logitsDesc.scalarType)")
    }

    public convenience init(
        config: ModelConfig,
        modelURL: URL,
        options: EngineOptions = EngineOptions()
    ) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(config: config, preparedModel: preparedModel, options: options)
    }

    // MARK: - Core forward (full re-encode of the running sequence)

    fileprivate func forwardLogits(for sequence: [Int32]) async throws -> [LogitsScalarType] {
        let L = sequence.count
        guard L > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot run on empty sequence")
        }
        guard L <= maxLen else {
            throw InferenceRuntimeError.contextLengthExceeded(L, maxLen)
        }

        let seqLen = dynamicSeq ? L : maxLen
        if dynamicSeq {
            inputIdsArray = NDArray(descriptor: inputIdsDescriptor.resolvingDynamicDimensions([1, seqLen]))
            positionIdsArray = NDArray(descriptor: positionIdsDescriptor.resolvingDynamicDimensions([1, seqLen]))
            logitsArray = NDArray(
                descriptor: logitsDescriptor.resolvingDynamicDimensions([1, seqLen, config.vocabSize]))
        }

        fillNDArray(&inputIdsArray, as: Int32.self, count: seqLen) { i in
            i < L ? sequence[i] : 0
        }
        fillNDArray(&positionIdsArray, as: Int32.self, count: seqLen) { Int32($0) }

        // Feed fresh zeroed caches each call; the graph fills them from scratch
        // within the pass. Caches may be bound as inputs or as state.
        var inputs: [String: NDArray] = [
            inputIdsName: inputIdsArray,
            positionIdsName: positionIdsArray,
        ]
        func makeCache(_ name: String) -> NDArray {
            let desc = cacheDescriptors[name]!
            let resolved = desc.shape.contains(where: { $0 < 0 })
                ? desc.resolvingDynamicDimensions(desc.shape.map { $0 < 0 ? seqLen : $0 })
                : desc
            var buf = NDArray(descriptor: resolved)
            _ = buf.mutableRawView()
            return buf
        }
        for name in cacheInputNames {
            inputs[name] = makeCache(name)
        }

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logitsArray, for: logitsName)

        // State buffers need stable storage for the non-escapable MutableViews,
        // so branch on the 0 (input-bound) vs 2 (key/value) state cases.
        if cacheStateNames.isEmpty {
            _ = try await function.run(
                inputs: inputs,
                states: InferenceFunction.MutableViews(),
                outputViews: consume outputViews
            )
        } else if cacheStateNames.count == 2 {
            var keyBuf = makeCache(cacheStateNames[0])
            var valBuf = makeCache(cacheStateNames[1])
            var states = InferenceFunction.MutableViews()
            states.insert(&keyBuf, for: cacheStateNames[0])
            states.insert(&valBuf, for: cacheStateNames[1])
            _ = try await function.run(
                inputs: inputs,
                states: consume states,
                outputViews: consume outputViews
            )
        } else {
            throw InferenceRuntimeError.invalidState(
                "Recompute engine supports 0 or 2 state caches, got \(cacheStateNames.count)")
        }

        // logits is [1, seqLen, vocab]; return the last real token's row.
        let v = config.vocabSize
        let full = readNDArray(logitsArray, as: LogitsScalarType.self, count: seqLen * v)
        let start = (L - 1) * v
        return Array(full[start..<(start + v)])
    }

    // MARK: - Generate (primary API)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        GenerationSequence(
            engine: self,
            input: input,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions
        )
    }

    // MARK: - Lifecycle

    private func drain() {
        var attempts = 0
        while generating.withLock({ $0 }) {
            attempts += 1
            if attempts > 5000 {
                fatalError("Recompute engine drain() timeout — generation Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    public func reset() {
        drain()
    }

    public func cleanup() {
        CLILogger.log("CoreAI recompute engine cleanup complete")
    }
}

// MARK: - Generation Sequence

extension CoreAIRecomputeEngine {
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let engine: CoreAIRecomputeEngine
        let input: [CoreAIRecomputeEngine.TokenId]
        let samplingConfiguration: SamplingConfiguration
        let inferenceOptions: InferenceOptions

        let stopReasonStore = StopReasonStore()
        public var stopReason: StopReason? { stopReasonStore.stopReason }
        public func setStopReason(_ reason: StopReason) { stopReasonStore.set(reason) }

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                engine: engine,
                input: input,
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore
            )
        }
    }
}

extension CoreAIRecomputeEngine.GenerationSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: CoreAIRecomputeEngine
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [CoreAIRecomputeEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore

        // The full running sequence (prompt + everything generated so far).
        private var sequence: [CoreAIRecomputeEngine.TokenId]
        private var step: Int = 0
        private var didAcquireLock: Bool = false
        private var finished: Bool = false

        init(
            engine: CoreAIRecomputeEngine,
            input: [CoreAIRecomputeEngine.TokenId],
            samplingConfiguration: SamplingConfiguration,
            inferenceOptions: InferenceOptions,
            stopReasonStore: StopReasonStore
        ) {
            self.engine = engine
            self.samplingConfiguration = samplingConfiguration
            self.returnsLogits = inferenceOptions.includeLogits
            self.forcedContinuation = inferenceOptions.forcedContinuation
            self.stopReasonStore = stopReasonStore
            self.sequence = input
            if let forced = inferenceOptions.forcedContinuation {
                self.maxTokens = forced.count
            } else {
                self.maxTokens = Swift.min(
                    inferenceOptions.maxTokens ?? Int.max,
                    Swift.max(0, engine.maxLen - input.count)
                )
            }
        }

        deinit {
            if didAcquireLock {
                engine.generating.withLock { $0 = false }
            }
        }

        public func next() async throws -> InferenceOutput? {
            if finished { return nil }

            if !didAcquireLock {
                engine.generating.withLock { $0 = true }
                didAcquireLock = true
            }

            guard step < maxTokens else {
                stopReasonStore.setIfUnset(.maxTokens)
                finishAndRelease()
                return nil
            }
            guard sequence.count <= engine.maxLen else {
                stopReasonStore.setIfUnset(.maxTokens)
                finishAndRelease()
                return nil
            }

            do {
                try Task.checkCancellation()

                let logitBuffer = try await engine.forwardLogits(for: sequence)

                let nextToken: Int32
                if let forced = forcedContinuation {
                    nextToken = forced[step]
                } else {
                    var mutableLogits = logitBuffer
                    nextToken = samplingConfiguration.fallbackSampler(from: &mutableLogits)
                }

                sequence.append(nextToken)
                step += 1

                return InferenceOutput(
                    tokenId: nextToken,
                    logits: returnsLogits ? logitBuffer : nil
                )
            } catch is CancellationError {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                finishAndRelease()
                throw error
            }
        }

        private func finishAndRelease() {
            guard !finished else { return }
            finished = true
            if didAcquireLock {
                engine.generating.withLock { $0 = false }
                didAcquireLock = false
            }
        }
    }
}
