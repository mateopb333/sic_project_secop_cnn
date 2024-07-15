import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_model(model, histories, X_test, y_test):
    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)
    logging.info(f"Loss: {loss}")
    logging.info(f"Accuracy: {accuracy}")

    # Promediar los historiales
    avg_history = {
        'accuracy': np.mean([h['accuracy'] for h in histories], axis=0),
        'val_accuracy': np.mean([h['val_accuracy'] for h in histories], axis=0),
        'loss': np.mean([h['loss'] for h in histories], axis=0),
        'val_loss': np.mean([h['val_loss'] for h in histories], axis=0)
    }

    # Graficar la precisión y la pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(avg_history['accuracy'], label='Train Accuracy')
    plt.plot(avg_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión del Modelo')
    plt.savefig('model_accuracy.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_history['loss'], label='Train Loss')
    plt.plot(avg_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida del Modelo')
    plt.savefig('model_loss.png')
    plt.close()

    # Matriz de confusión
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusión')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Reporte de Clasificación:\n{classification_report(y_test, y_pred)}")

    # Curvas ROC
    n_classes = len(np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC por clase')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()

def predict_example(model, tokenizer, example_text, maxlen):
    example_seq = tokenizer.texts_to_sequences([example_text])
    example_pad = pad_sequences(example_seq, maxlen=maxlen)
    prediction = model.predict(example_pad)
    return prediction

