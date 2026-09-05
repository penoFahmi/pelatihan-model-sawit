import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Persiapan Lingkungan dan Import Pustaka\n",
    "Pipeline V2 dengan arsitektur MobileNetV2 yang dioptimasi secara ekstensif untuk dataset berukuran kecil (penambahan Dropout, Dense Layer, standar pra-pemrosesan Keras, dan implementasi Callbacks/EarlyStopping)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import splitfolders\n",
    "import albumentations as A\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
    "\n",
    "# Konfigurasi GPU (Opsional)\n",
    "gpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "if gpus:\n",
    "    try:\n",
    "        for gpu in gpus:\n",
    "            tf.config.experimental.set_memory_growth(gpu, True)\n",
    "        print(\"GPU Siap dengan memory growth aktif.\")\n",
    "    except RuntimeError as e:\n",
    "        print(e)\n",
    "else:\n",
    "    print(\"Menggunakan Komputasi CPU.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Partisi Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "input_dir = 'dataset'\n",
    "output_dir = 'dataset_split'\n",
    "\n",
    "# Memisahkan folder utama menjadi Train (70%), Val (15%), Test (15%)\n",
    "if not os.path.exists(output_dir):\n",
    "    if os.path.exists(input_dir) and len(os.listdir(input_dir)) > 0:\n",
    "        print(\"Mengeksekusi pemisahan dataset...\")\n",
    "        splitfolders.ratio(input_dir, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15))\n",
    "        print(\"Pemisahan selesai.\")\n",
    "    else:\n",
    "        print(\"ERROR: Folder 'dataset' kosong atau tidak ada.\")\n",
    "else:\n",
    "    print(\"Folder 'dataset_split' sudah ada. Lanjut ke proses berikutnya.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Pra-pemrosesan Data (ImageDataGenerator + Albumentations)\n",
    "Kita mengintegrasikan `Albumentations` langsung ke dalam parameter `preprocessing_function` dari Keras. Hal ini menjamin efisiensi memori, ditambah pemanggilan fungsi wajib `preprocess_input` khusus dari MobileNetV2 untuk memetakan piksel ke rentang `[-1, 1]`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Definisi Augmentasi Fisik & Warna\n",
    "transform = A.Compose([\n",
    "    A.Rotate(limit=30, p=0.5),\n",
    "    A.HorizontalFlip(p=0.5),\n",
    "    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),\n",
    "    A.Affine(scale=(0.8, 1.2), p=0.5), # Zoom\n",
    "    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),\n",
    "])\n",
    "\n",
    "# Fungsi Wrapper untuk ImageDataGenerator (Data Latih)\n",
    "def aug_fn(image):\n",
    "    # Keras memberikan gambar dalam bentuk numpy array float32 [0, 255]\n",
    "    data = {\"image\": image.astype(np.uint8)}\n",
    "    aug_data = transform(**data)\n",
    "    aug_img = aug_data[\"image\"].astype(np.float32)\n",
    "    # Sangat Krusial: Transformasi ke skala [-1, 1] standar MobileNetV2\n",
    "    return preprocess_input(aug_img) \n",
    "\n",
    "# Fungsi Wrapper tanpa augmentasi (Data Validasi & Uji)\n",
    "def no_aug_fn(image):\n",
    "    return preprocess_input(image.astype(np.float32))\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "IMG_SIZE = (224, 224)\n",
    "\n",
    "train_datagen = ImageDataGenerator(preprocessing_function=aug_fn)\n",
    "val_test_datagen = ImageDataGenerator(preprocessing_function=no_aug_fn)\n",
    "\n",
    "train_gen = train_datagen.flow_from_directory(\n",
    "    os.path.join(output_dir, 'train'),\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "val_gen = val_test_datagen.flow_from_directory(\n",
    "    os.path.join(output_dir, 'val'),\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "test_gen = val_test_datagen.flow_from_directory(\n",
    "    os.path.join(output_dir, 'test'),\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "# Mengambil list kelas\n",
    "classes = list(train_gen.class_indices.keys())\n",
    "print(\"\\nKelas Terdeteksi:\", classes)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Arsitektur Model Baru & Callbacks Pelatihan\n",
    "Penambahan `Dense(256)` dan `Dropout(0.5)` mengatasi ketidakmampuan model dasar membaca fitur spesifik sawit. Ditambah penerapan `EarlyStopping` dan `ModelCheckpoint` agar hasil panen terbaik di-save otomatis tanpa khawatir rusak di epoch terakhir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\n",
    "\n",
    "if classes:\n",
    "    # 1. Base Model (Frozen)\n",
    "    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "    base_model.trainable = False\n",
    "\n",
    "    # 2. Modifikasi Top Layers (Telah Diperkuat)\n",
    "    x = base_model.output\n",
    "    x = GlobalAveragePooling2D()(x)\n",
    "    x = Dense(256, activation='relu')(x) # Hidden layer baru untuk kapabilitas berpikir\n",
    "    x = Dropout(0.5)(x)                  # Mencegah overfitting secara ekstrim\n",
    "    predictions = Dense(len(classes), activation='softmax')(x)\n",
    "    \n",
    "    model = Model(inputs=base_model.input, outputs=predictions)\n",
    "\n",
    "    # 3. Kompilasi Tahap 1 (Learning rate 1e-3 agar top layer cepat belajar)\n",
    "    model.compile(optimizer=Adam(learning_rate=1e-3), \n",
    "                  loss='categorical_crossentropy', \n",
    "                  metrics=['accuracy'])\n",
    "\n",
    "    # 4. Kalkulasi Class Weights Otomatis untuk Keadilan Pembelajaran\n",
    "    train_labels = train_gen.classes\n",
    "    class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)\n",
    "    class_weight_dict = {i: weight for i, weight in enumerate(class_weights_arr)}\n",
    "    print(\"Class Weights:\", class_weight_dict)\n",
    "\n",
    "    # 5. Persiapan Callbacks Keamanan\n",
    "    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)\n",
    "    early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)\n",
    "    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)\n",
    "    callbacks_list = [checkpoint, early_stop, reduce_lr]\n",
    "\n",
    "    print(\"\\n--- Mulai Tahap 1 (Feature Extraction / Frozen Base) ---\")\n",
    "    # Pelatihan Berjalan\n",
    "    history_phase1 = model.fit(\n",
    "        train_gen,\n",
    "        validation_data=val_gen,\n",
    "        epochs=20,\n",
    "        class_weight=class_weight_dict,\n",
    "        callbacks=callbacks_list\n",
    "    )\n",
    "else:\n",
    "    print(\"Dataset kosong.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Fine Tuning Lanjutan (Tahap 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if classes:\n",
    "    # Buka seluruh gembok layer\n",
    "    base_model.trainable = True\n",
    "    # Kunci kembali layer-layer dasar (kecuali 30 layer teratas)\n",
    "    for layer in base_model.layers[:-30]:\n",
    "        layer.trainable = False\n",
    "\n",
    "    # Kompilasi Ulang dengan LR super lambat (1e-5)\n",
    "    model.compile(optimizer=Adam(learning_rate=1e-5), \n",
    "                  loss='categorical_crossentropy', \n",
    "                  metrics=['accuracy'])\n",
    "\n",
    "    print(\"\\n--- Mulai Tahap 2 (Fine-Tuning 30 Layers Atas) ---\")\n",
    "    # Pelatihan Tahap 2 Berjalan\n",
    "    history_phase2 = model.fit(\n",
    "        train_gen,\n",
    "        validation_data=val_gen,\n",
    "        epochs=30,\n",
    "        class_weight=class_weight_dict,\n",
    "        callbacks=callbacks_list\n",
    "    )\n",
    "else:\n",
    "    pass\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Kuantisasi ke Format Android (TensorFlow Lite)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if classes and os.path.exists('best_model.keras'):\n",
    "    print(\"Memuat bobot terbaik dan Melakukan Kuantisasi (Float16)...\")\n",
    "    # Karena EarlyStopping me-restore best weights, 'model' di memori adalah yang terbaik.\n",
    "    # Anda juga bisa secara eksplisit me-load best_model.keras\n",
    "    # model.load_weights('best_model.keras')\n",
    "    \n",
    "    converter = tf.lite.TFLiteConverter.from_keras_model(model)\n",
    "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n",
    "    converter.target_spec.supported_types = [tf.float16]\n",
    "\n",
    "    # Menjalankan Kuantisasi\n",
    "    tflite_quant_model = converter.convert()\n",
    "    tflite_filepath = \"mobilenetv2_sawit_fp16.tflite\"\n",
    "    with open(tflite_filepath, \"wb\") as f:\n",
    "        f.write(tflite_quant_model)\n",
    "    print(f\"Model TFLite tersimpan sebagai {tflite_filepath}\")\n",
    "else:\n",
    "    print(\"Menunggu hasil pelatihan untuk dikuantisasi.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Evaluasi Akhir (Data Uji 15%)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'history_phase1' in locals() and 'history_phase2' in locals():\n",
    "    def plot_curves(h1, h2):\n",
    "        acc = h1.history['accuracy'] + h2.history['accuracy']\n",
    "        val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']\n",
    "        loss = h1.history['loss'] + h2.history['loss']\n",
    "        val_loss = h1.history['val_loss'] + h2.history['val_loss']\n",
    "        epochs_range = range(1, len(acc) + 1)\n",
    "        plt.figure(figsize=(12, 4))\n",
    "        plt.subplot(1, 2, 1)\n",
    "        plt.plot(epochs_range, acc, label='Train Acc', linewidth=2)\n",
    "        plt.plot(epochs_range, val_acc, label='Val Acc', linewidth=2)\n",
    "        plt.legend(loc='lower right')\n",
    "        plt.title('Dinamika Akurasi (V2)')\n",
    "        plt.subplot(1, 2, 2)\n",
    "        plt.plot(epochs_range, loss, label='Train Loss', linewidth=2)\n",
    "        plt.plot(epochs_range, val_loss, label='Val Loss', linewidth=2)\n",
    "        plt.legend(loc='upper right')\n",
    "        plt.title('Dinamika Loss (V2)')\n",
    "        plt.show()\n",
    "\n",
    "    plot_curves(history_phase1, history_phase2)\n",
    "\n",
    "if classes:\n",
    "    print(\"\\n--- Mengukur Metrik di Dataset Uji ---\")\n",
    "    test_gen.reset() # Mereset urutan pointer sebelum memprediksi\n",
    "    y_pred_probs = model.predict(test_gen)\n",
    "    y_pred = np.argmax(y_pred_probs, axis=1)\n",
    "    y_true = test_gen.classes\n",
    "\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)\n",
    "    plt.title('Confusion Matrix - Data Uji')\n",
    "    plt.show()\n",
    "\n",
    "    print(\"\\n=======================================================\")\n",
    "    print(\"LAPORAN AKHIR V2 (CLASSIFICATION REPORT)\")\n",
    "    print(\"=======================================================\")\n",
    "    print(classification_report(y_true, y_pred, target_names=classes))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('pelatihan_mobilenetv2_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("V2 Notebook Created Successfully.")
