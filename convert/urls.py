from django.urls import path
from .views import ConvertImagesView

urlpatterns = [
    path("", ConvertImagesView.as_view()),
]
