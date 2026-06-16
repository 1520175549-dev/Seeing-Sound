import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResSiren(nn.Module):
    def __init__(self):
        super(ResSiren, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        self.blk1 = ResBlk(32, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=1)
        self.blk3 = ResBlk(128, 256, stride=1)
        self.blk4 = ResBlk(256, 512, stride=1)
        self.outlayer = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        return self.outlayer(x)


criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResSiren().to(DEVICE)
    MODEL_PATH = "ressiren_weights.pth"
    model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 1, 128, 126).to(DEVICE)

    class ResSirenForWeb(nn.Module):
        def __init__(self, trained_model):
            super(ResSirenForWeb, self).__init__()
            self.model = trained_model
        def forward(self, x):
            logits = self.model(x)
            probs = F.softmax(logits, dim = 1)
            label = torch.argmax(probs, dim = 1)
            return probs, label

    web_model = ResSirenForWeb(model)
    dummy_input = torch.randn(1, 1, 128, 126).to(DEVICE)

    torch.onnx.export(
        web_model,
        dummy_input,
        "ressiren.onnx",
        input_names = ['input'],
        output_names = ['probabilities', 'label'],
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'label': {0: 'batch_size'}
        },
        opset_version = 11
    )

    import onnx
    model_onnx = onnx.load("ressiren.onnx")
    onnx.save_model(model_onnx, "ressiren_single.onnx", save_as_external_data=False)

    import os
    print(f"Done! Size: {os.path.getsize('ressiren_single.onnx') / 1024 / 1024:.1f} MB")
