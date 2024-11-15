from django.http import FileResponse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions

from .serializers import ConvertImagesSerializer

from modules.faker.adversarial_image import generate_adversarial_image


# Create your views here.
class ConvertImagesView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, format=None):
        return Response("Send with POST", status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, format=None):
        serializer = ConvertImagesSerializer(data=request.data)
        if serializer.is_valid():
            image = bytes(request.data["image"].file.read())
            files = generate_adversarial_image(image)
            files.seek(0)

            return FileResponse(files, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
