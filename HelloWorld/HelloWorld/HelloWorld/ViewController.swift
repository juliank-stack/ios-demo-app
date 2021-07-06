import UIKit

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    struct PixelData {
        var a: UInt8
        var r: UInt8
        var g: UInt8
        var b: UInt8
    }
    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "cpu", ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()

    private lazy var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        let image = UIImage(named: "tpose.jpeg")!
        imageView.image = image
        let resizedImage = image.resized(to: CGSize(width: 192, height: 256))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
            return
        }
        let swiftArray: [Float] = outputs.compactMap({ $0 as? Float })
        for hm in 1...16 {
            var pixels = [PixelData]()
            let slice: ArraySlice<Float>
            let overBorder = hm * 3072 - 1
            let underBorder = overBorder - 3071
            slice = swiftArray[underBorder...overBorder]
            let max = slice.max()
            let min = slice.min()
            
            for x in underBorder...overBorder{
                if slice[x] == max {
                    pixels.append(PixelData(a: 255,  r:   255   ,g:0,b:0))
                }
                else {
                    pixels.append(PixelData(a: 0,  r:   0   ,g:0,b:0))
                }
                //let val = UInt8(slice[x]*255/max!)
                //pixels.append(PixelData(a: UInt8((slice[x]-min!)*255/(max! - min!)),  r:   0   ,g:0,b:255))
                if max! > 0.6 {
                    print("something")
                }
            }
            
            let image = imageFromARGB32Bitmap(pixels: pixels, width: 48, height: 64)
            let hmView = UIImageView(image: image!)
            hmView.frame = CGRect(x: 0, y: 0, width: 48*4, height: 64*4)
            //imageView.transform = imageView.transform.rotated(by: .pi / 2)
            resultView.addSubview(hmView)
        }
        func imageFromARGB32Bitmap(pixels: [PixelData], width: Int, height: Int) -> UIImage? {
            guard width > 0 && height > 0 else { return nil }
            guard pixels.count == width * height else { return nil }

            let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
            let bitsPerComponent = 8
            let bitsPerPixel = 32

            var data = pixels // Copy to mutable []
            guard let providerRef = CGDataProvider(data: NSData(bytes: &data,
                                    length: data.count * MemoryLayout<PixelData>.size)
                )
                else { return nil }

            guard let cgim = CGImage(
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bitsPerPixel: bitsPerPixel,
                bytesPerRow: width * MemoryLayout<PixelData>.size,
                space: rgbColorSpace,
                bitmapInfo: bitmapInfo,
                provider: providerRef,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
                )
                else { return nil }

            return UIImage(cgImage: cgim)
        }
        
//        let zippedResults = zip(labels.indices, outputs)
//        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
//        var text = ""
//        for result in sortedResults {
//            text += "\u{2022} \(labels[result.0]) \n\n"
//        }

    }
}
