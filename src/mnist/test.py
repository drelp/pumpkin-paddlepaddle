import paddle
from paddle.vision.transforms import Compose, Normalize

if __name__ == "__main__":
    print(paddle.__version__)
    transform = Compose([Normalize(mean=[127.5],
                                   std=[127.5],
                                   data_format='CHW')])
    # 使用transform对数据集做归一化
    print('download training data and load training data')
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    print('load finished')