from d2lzh import d2lzh as d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import os
import numpy as np
import matplotlib.pyplot as plt


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
[0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, self.get_blk(i))
            setattr(self, 'cls_%d' % i, self.cls_predictor(num_anchors,num_classes))
            setattr(self, 'bbox_%d' % i, self.bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # reshape函数中的0表⽰保持批量⼤⼩不变
        return (nd.concat(*anchors, dim=1),
            self.concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)), self.concat_preds(bbox_preds))

    # 类别预测
    def cls_predictor(self, num_anchors, num_classes):
        return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                         padding=1)

    # 边界预测
    def bbox_predictor(self, num_anchors):
        return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

    # # 前向传播
    # def forward_ini(x, block):
    #     block.initialize()
    #     return block(x)

    # 平滑
    def flatten_pred(self, pred):
        return pred.transpose((0, 2, 3, 1)).flatten()

    # 把平滑后连在一起
    def concat_preds(self, preds):
        return nd.concat(*[self.flatten_pred(p) for p in preds], dim=1)

    # 尺度变化，卷积加池化
    def down_sample_blk(self, num_channels):
        blk = nn.Sequential()
        for _ in range(2):
            blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                    nn.BatchNorm(in_channels=num_channels),
                    nn.Activation('relu'))
        blk.add(nn.MaxPool2D(2))
        return blk

    # 设置三次尺度减半
    def base_net(self):
        blk = nn.Sequential()
        for num_filters in [16, 32, 64]:
            blk.add(self.down_sample_blk(num_filters))
        return blk

    # 上面三个模块整合
    def get_blk(self, i):
        if i == 0:
            blk = self.base_net()
        elif i == 4:
            blk = nn.GlobalMaxPool2D()
        else:
            blk = self.down_sample_blk(128)
        return blk

    # 前向传播
    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)




class TinySSD_train():
    def __init__(self, batch_size, **kwargs):
        super(TinySSD_train, self).__init__(**kwargs)
        self.batch_size = batch_size

        self.train_iter, _ = self.load_data_pikachu(self.batch_size)
        self.ctx, self.net = d2l.try_gpu(), TinySSD(num_classes=1)
        self.net.initialize(init=init.Xavier(), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd',
                                {'learning_rate': 0.2, 'wd': 5e-4})
        self.cls_loss = gloss.SoftmaxCrossEntropyLoss()
        self.bbox_loss = gloss.L1Loss()
    def load_data_pikachu(self, batch_size, edge_size=256):  # edge_size：输出图像的宽和⾼
        data_dir = './data/pikachu'
        os.makedirs(data_dir, exist_ok=True)
        train_iter = image.ImageDetIter(
            path_imgrec=os.path.join(data_dir, 'train.rec'),
            path_imgidx=os.path.join(data_dir, 'train.idx'),
            batch_size=batch_size,
            data_shape=(3, edge_size, edge_size),  # 输出图像的形状
            shuffle=True,  # 以随机顺序读取数据集
            rand_crop=1,  # 随机裁剪的概率为1
            min_object_covered=0.95, max_attempts=200)
        val_iter = image.ImageDetIter(
            path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
            data_shape=(3, edge_size, edge_size), shuffle=False)
        return train_iter, val_iter




    def calc_loss(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls = self.cls_loss(cls_preds, cls_labels)
        bbox = self.bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox


    def cls_eval(self, cls_preds, cls_labels):
        # 由于类别预测结果放在最后⼀维，argmax需要指定最后⼀维
        return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()


    def bbox_eval(self, bbox_preds, bbox_labels, bbox_masks):
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    # def cls_eval(cls_preds, cls_labels):
    #     # Because the category prediction results are placed in the final
    #     # dimension, argmax must specify this dimension
    #     return float((cls_preds.argmax(axis=-1).astype(cls_labels.dtype) == cls_labels).sum())
    #
    # def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    #     return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

    def train(self):
        for epoch in range(20):
            acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
            self.train_iter.reset()  # 从头读取数据
            start = time.time()
            for batch in self.train_iter:
                X = batch.data[0].as_in_context(self.ctx)
                Y = batch.label[0].as_in_context(self.ctx)
                with autograd.record():
                    # ⽣成多尺度的锚框，为每个锚框预测类别和偏移量
                    anchors, cls_preds, bbox_preds = self.net(X)
                    # 为每个锚框标注类别和偏移量
                    bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                        anchors, Y, cls_preds.transpose((0, 2, 1)))
                    # 根据类别和偏移量的预测和标注值计算损失函数
                    l = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                                  bbox_masks)
                l.backward()
                self.trainer.step(self.batch_size)
                acc_sum += self.cls_eval(cls_preds, cls_labels)
                n += cls_labels.size
                mae_sum += self.bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                m += bbox_labels.size
            if (epoch + 1) % 5 == 0:
                print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                    epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))

    def predict(self, X):
        anchors, cls_preds, bbox_preds = self.net(X.as_in_context(self.ctx))

        cls_probs = cls_preds.softmax().transpose((0, 2, 1))
        output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
        return output[0, idx]

    def display(self, img, output, threshold):
        d2l.set_figsize((5, 5))
        fig = d2l.plt.imshow(img.asnumpy())
        for row in output:
            score = row[1].asscalar()
            if score < threshold:
                continue
            h, w = img.shape[0:2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

if __name__ == '__main__':
    img = image.imread('./data/pikachu/val/images/90.png')
    feature = image.imresize(img, 256, 256).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

    model = TinySSD_train(batch_size=32)
    model.train()
    output = model.predict(X)
    model.display(img, output, threshold=0.3)
    # Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
    # Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
    # print(np.shape(concat_preds([Y1, Y2])))