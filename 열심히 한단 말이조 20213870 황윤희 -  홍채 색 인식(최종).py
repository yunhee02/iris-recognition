import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 데이터 경로 설정
train_data_dir = "C:\\Users\\82103\\Desktop\\eyes"
valid_data_dir = "C:\\Users\\82103\\Desktop\\eyes img"

# 이미지 크기와 배치 크기 설정
image_size = (64, 64)
batch_size = 32

# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # 이미지 회전 범위 설정
    width_shift_range=0.1,  # 가로 방향으로 이동할 범위 설정
    height_shift_range=0.1,  # 세로 방향으로 이동할 범위 설정
    shear_range=0.2,  # 이미지 전단 변환 범위 설정
    zoom_range=0.2,  # 이미지 확대/축소 범위 설정
    horizontal_flip=True  # 수평으로 뒤집기 설정
)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# 모델 구조 정의
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),  # 배치 정규화 레이어 추가
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # 드롭아웃 레이어 추가
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
epochs = 20  # 더 많은 에포크 수로 훈련
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=valid_generator)

# 모델 평가
loss, accuracy = model.evaluate(valid_generator)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# 손실과 정확도 그래프 그리기
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.plot(epochs, val_accuracy, 'm', label='Validation Accuracy')
plt.title('Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()
