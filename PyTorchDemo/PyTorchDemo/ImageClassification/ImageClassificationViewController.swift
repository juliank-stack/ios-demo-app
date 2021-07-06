import AVFoundation
import UIKit


class ImageClassificationViewController: ViewController {
    @IBOutlet var cameraView: CameraPreviewView!
    @IBOutlet var bottomView: ImageClassificationResultView!
    @IBOutlet var benchmarkLabel: UILabel!
    @IBOutlet var indicator: UIActivityIndicatorView!
    private var predictor = ImagePredictor()
    private var cameraController = CameraController()
    private let delayMs: Double = 500
    private var prevTimestampMs: Double = 0.0
    private var firstTime = 0
    struct PixelData {
        var a: UInt8
        var r: UInt8
        var g: UInt8
        var b: UInt8
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView.config(resultCount: 3)
        cameraController.configPreviewLayer(cameraView)
        cameraController.videoCaptureCompletionBlock = { [weak self] normalizedBuffer, error in
            guard let strongSelf = self else { return }
            if error != nil {
                strongSelf.showAlert(error)
                return
            }
            guard let pixelBuffer = normalizedBuffer else { return }
            let currentTimestamp = CACurrentMediaTime()
            if (currentTimestamp - strongSelf.prevTimestampMs) * 1000 <= strongSelf.delayMs { return }
            strongSelf.prevTimestampMs = currentTimestamp
            if let results = try? strongSelf.predictor.predict(pixelBuffer, resultCount: 3) {
                DispatchQueue.main.async {
                    strongSelf.indicator.isHidden = true
                    strongSelf.bottomView.isHidden = true
                    strongSelf.benchmarkLabel.isHidden = false
                    
                    

                    //let red = PixelData(a: 100, r: 255, g: 0, b: 0)
                    //let green = PixelData(a: 100, r: 0, g: 255, b: 0)
                    //let blue = PixelData(a: 100, r: 0, g: 0, b: 255)
                    let swiftArray: [Float] = results.0.compactMap({ $0 as? Float })
                    for hm in 5...5 {
                        var pixels = [PixelData]()
                        let slice: ArraySlice<Float>
                        let overBorder = hm * 3072 - 1
                        let underBorder = overBorder - 3071
                        slice = swiftArray[underBorder...overBorder]
                        let max = slice.max()
                        let min = slice.min()
                        
                        for x in underBorder...overBorder{
                            if slice[x] == max {
                                //pixels.append(PixelData(a: 255,  r:   255   ,g:0,b:0))
                            }
                            else {
                                //pixels.append(PixelData(a: 0,  r:   0   ,g:0,b:0))
                            }
                            //let val = UInt8(slice[x]*255/max!)
                            pixels.append(PixelData(a: UInt8((slice[x]-min!)*255/(max! - min!)),  r:   255   ,g:0,b:0))
                            if max! > 0.6 {
                                print("something")
                            }
                        }
                        
                        let image = self!.imageFromARGB32Bitmap(pixels: pixels, width: 48, height: 64)
                        let imageView = UIImageView(image: image!)
                        imageView.frame = CGRect(x: 100, y: 100, width: 480, height: 640)
                        //imageView.transform = imageView.transform.rotated(by: .pi / 2)
                        strongSelf.cameraView.addSubview(imageView)
                    }
                    
                    if self!.firstTime == 1 {
                        //print("delete")
                        strongSelf.cameraView.subviews[4].removeFromSuperview()
//                        strongSelf.cameraView.subviews[5].removeFromSuperview()
//                        strongSelf.cameraView.subviews[6].removeFromSuperview()
//                        strongSelf.cameraView.subviews[7].removeFromSuperview()
//                        strongSelf.cameraView.subviews[8].removeFromSuperview()
//                        strongSelf.cameraView.subviews[9].removeFromSuperview()
                        
                        
                    }
                    self!.firstTime = 1
                    

                    
                    strongSelf.benchmarkLabel.text = String(format: "%.2f",results.1)
                    
                    //strongSelf.bottomView.update(results :results.0)
                }
                
            }
        }
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
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(true, animated: false)
        cameraController.startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraController.stopSession()
    }

    @IBAction func onInfoBtnClicked(_: Any) {
        VisionModelCard.show()
    }

    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
