# 初學者的 TensorFlow 2.0教程


##  將 TensorFlow 導入到程式碼

```python
import tensorflow as tf
```

## 加入Minist數據庫

載入並準備 MNIST 数据集。將樣本數據從整数轉换為浮点数：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step

```
## 建構機器學習模型

透過堆疊層來建構 tf.keras.Sequential 模型。

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

對於每個樣本，模型都會傳回一個包含 logits 或 log-odds 分數的向量，每個類別一個。

```python
predictions = model(x_train[:1]).numpy()
predictions
     
array([[ 0.44744357,  0.05407905, -0.41832733, -0.3854866 ,  0.09117224,
        -0.24533486,  0.32785457, -0.12109746,  0.2927832 , -0.66886187]],
      dtype=float32)
```

tf.nn.softmax 函數將這些 logits 轉換為每個類別的機率：

```python
array([[0.15702088, 0.10595497, 0.06606293, 0.06826851, 0.109959  ,
        0.0785394 , 0.13932228, 0.08892895, 0.13452074, 0.05142237]],
      dtype=float32)
```

使用 losses.SparseCategoricalCrossentropy 為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每個樣本傳回一個標量損失。

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

此損失等於 true 類別的負對數機率：如果模型確定類別正確，則損失為零。

這個未經訓練的模型給出的機率接近隨機（每個類別為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3。

```python
loss_fn(y_train[:1], predictions).numpy()
     
2.5441551
```

在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 
將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
     
```

## 訓練並評估模型

使用 Model.fit 方法調整您的模型參數並最小化損失：

```python
model.fit(x_train, y_train, epochs=5)
     
Epoch 1/5
1875/1875 [==============================] - 9s 4ms/step - loss: 0.2995 - accuracy: 0.9119
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1460 - accuracy: 0.9556
Epoch 3/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.1122 - accuracy: 0.9663
Epoch 4/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0909 - accuracy: 0.9718
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0780 - accuracy: 0.9752
<keras.src.callbacks.History at 0x7cd51a028430>
```

Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。

```python
model.evaluate(x_test,  y_test, verbose=2)
     
313/313 - 1s - loss: 0.0773 - accuracy: 0.9740 - 736ms/epoch - 2ms/step
[0.0773211270570755, 0.9739999771118164]
```

現在，這個照片分類器的準確度已經接近 98%。

如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

```python

probability_model(x_test[:5])
     
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[2.18498410e-08, 5.96625194e-09, 5.45496960e-07, 2.06778168e-05,
        3.21207037e-12, 3.27697940e-08, 8.33453146e-15, 9.99977350e-01,
        1.04099644e-08, 1.43630655e-06],
       [2.70749689e-09, 5.17076332e-05, 9.99935389e-01, 8.31654688e-06,
        3.11520655e-15, 3.82673306e-06, 2.86501830e-07, 7.64732092e-15,
        4.83221811e-07, 1.69338640e-14],
       [1.49924233e-07, 9.98896241e-01, 1.18878583e-04, 2.08440724e-06,
        1.20362961e-06, 6.22299740e-06, 7.80252867e-06, 7.17313960e-04,
        2.48640776e-04, 1.48377171e-06],
       [9.99921441e-01, 1.02781845e-08, 4.21633922e-06, 2.16699007e-07,
        2.40890058e-05, 4.41206566e-06, 4.28830099e-05, 1.88480300e-07,
        2.80243251e-09, 2.65526091e-06],
       [6.96401275e-06, 1.45653989e-09, 2.01319926e-06, 9.43925187e-08,
        9.95501339e-01, 2.42118335e-06, 1.08787599e-05, 6.44610327e-06,
        1.91044455e-06, 4.46777558e-03]], dtype=float32)>
```

## 參考文章:
https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn
