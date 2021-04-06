import tensorflow as tf
from dataset import MovielensDataset
from modeling import DCN
from new_modeling import tDCN


def main():
    dataset = MovielensDataset()
    cached_train, cached_test, vocabularies = dataset.get_data()
    # model = DCN(deep_layer_sizes=[192, 192], projection_dim=20, vocabularies=vocabularies)
    model = tDCN(vocabularies=vocabularies)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  metrics=tf.keras.metrics.RootMeanSquaredError('RMSE'),
                  loss=tf.keras.losses.MeanSquaredError())
    for i in range(50):
        print(i)
        model.fit(cached_train, epochs=1)
        model.evaluate(cached_test, return_dict=True)


if __name__ == '__main__':
    main()
