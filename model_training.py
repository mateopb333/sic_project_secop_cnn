import logging
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class ModelTrainer:
    def __init__(self, build_model_fn, X, y, vocab_size, embedding_dim, input_length, n_splits=5, epochs=20, batch_size=32):
        self.build_model_fn = build_model_fn
        self.X = X
        self.y = y
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.histories = []

    def train(self):
        logging.info('Iniciando el entrenamiento del modelo con k-fold cross-validation.')
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(kf.split(self.X)):
            logging.info(f'Entrenando el pliegue {fold + 1}/{self.n_splits}')

            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Crear una nueva instancia del modelo para cada pliegue
            model = self.build_model_fn(self.vocab_size, self.embedding_dim)

            # Calcular los pesos de clase
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: class_weights[i] for i in range(len(class_weights))}

            # Early stopping para evitar el sobreajuste
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weights,
                callbacks=[early_stopping]
            )

            self.histories.append(history.history)
            logging.info(f'Pliegue {fold + 1}/{self.n_splits} completado.')

        logging.info('Entrenamiento del modelo con k-fold cross-validation completado.')
        return self.histories
