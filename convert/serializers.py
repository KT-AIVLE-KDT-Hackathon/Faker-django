from rest_framework import serializers
from .models import ConvertImages


class ConvertImagesSerializer(serializers.Serializer):
    image = serializers.ImageField()
