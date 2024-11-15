import io
import tensorflow as tf
import numpy as np
from PIL import Image

from .fgsm_attack import apply_fgsm_attack
from .lpips_loss import LPIPSLossWithFGSM


def generate_adversarial_image(image: bytes) -> io.BytesIO:
    # 이미지 불러오기
    image = tf.convert_to_tensor(image)
    image = tf.image.decode_image(image, channels=3)  # 이미지 디코딩 (3채널 RGB)
    image = tf.cast(image, dtype=tf.float32) / 255.0  # 이미지를 [0, 1] 범위로 정규화
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가 (배치 크기 1)

    # FGSM 공격 적용
    lpips_fgsm_loss = LPIPSLossWithFGSM(trunk_network="alex", epsilon=0.01)
    adversarial_image, loss_value = apply_fgsm_attack(
        lpips_fgsm_loss, image, epsilon=0.01)
    if adversarial_image.shape[0] == 1:  # 배치 차원만 검사
        adversarial_image_numpy = tf.squeeze(
            adversarial_image).numpy()  # 배치 차원 제거

        img = np.clip(
            adversarial_image_numpy * 255,
            0,
            255).astype(np.uint8)  # 값 클리핑
        img = Image.fromarray(img)  # NumPy 배열을 PIL 이미지로 변환

        imagefile = io.BytesIO()
        img.save(imagefile, format='PNG')
        return imagefile
    else:
        print(
            f"Invalid shape for adversarial image: {adversarial_image.shape}")
    return None
