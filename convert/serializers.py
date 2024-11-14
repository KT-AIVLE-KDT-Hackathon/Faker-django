from rest_framework import serializers


class ConvertImagesSerializer(serializers.Serializer):
    image = serializers.ImageField()
