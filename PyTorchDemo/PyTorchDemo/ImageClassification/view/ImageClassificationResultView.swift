import UIKit

class ImageClassificationResultView: UIView {
    @IBOutlet var contentView: UIView!
    @IBOutlet var containerView: UIStackView!
    var itemViews: [ImageClassificationItemView] = []
    let colors = [0xE8492B, 0xC52E8B, 0x7C2BDE]
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }
    
    func commonInit() {
        Bundle.main.loadNibNamed("ImageClassificationResultView", owner: self, options: nil)
        contentView.setup(self)
    }

    func config(resultCount count: Int) {
        for index in 0 ..< count {
            let itemView = ImageClassificationItemView(frame: .zero)
            if index == 0 {
                itemView.resultLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
                itemView.scoreLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
            } else {
                itemView.resultLabel.font = UIFont.systemFont(ofSize: 14.0)
                itemView.scoreLabel.font = UIFont.systemFont(ofSize: 14.0)
            }
            itemView.gradientColor = UIColor(rgb: colors[index])
            containerView.addArrangedSubview(itemView)
            itemViews.append(itemView)
        }
    }
    
    func update(results: Int) {
//        for index in 0 ..< results.count {
//            let itemView = itemViews[index]
//            itemView.resultLabel.text = results[index].label
//            itemView.scoreLabel.text = String(format: "%.2f", results[index].score)
//        }
        let itemView = itemViews[1]
        
        itemView.resultLabel.text = String(results)
        itemView.scoreLabel.text = String(results)
    }
}
