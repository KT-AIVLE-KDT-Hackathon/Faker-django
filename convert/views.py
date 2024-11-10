from django.http import FileResponse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions

from .models import ConvertImages
from .serializers import ConvertImagesSerializer


# Create your views here.
class ConvertImagesView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, format=None):
        images = ConvertImages.objects.all()
        serializer = ConvertImagesSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = ConvertImagesSerializer(data=request.data)
        if serializer.is_valid():
            image = request.data["image"].file
            return FileResponse(image, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
