from django.urls import path
from .views import UploadTradesAPIView, TradesListAPIView, AnalyzeAPIView

urlpatterns = [
    path("upload/", UploadTradesAPIView.as_view(), name="api-upload"),
    path("trades/", TradesListAPIView.as_view(), name="api-trades"),
    path("analyze/", AnalyzeAPIView.as_view(), name="api-analyze"),
]
