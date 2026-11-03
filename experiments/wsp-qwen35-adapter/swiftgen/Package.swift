// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "wspgen",
    platforms: [.macOS("27.0")],
    targets: [
        .executableTarget(
            name: "wspgen",
            path: "Sources/wspgen",
            swiftSettings: [.enableExperimentalFeature("Lifetimes")]
        )
    ]
)
