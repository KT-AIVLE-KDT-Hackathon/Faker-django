from rest_framework import serializers
from .models import ConvertImages


class ConvertImagesSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConvertImages
        fields = '__all__'
