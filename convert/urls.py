from django.urls import path
from .views import ConvertImagesView

urlpatterns = [
    path("convert", ConvertImagesView.as_view()),
]
