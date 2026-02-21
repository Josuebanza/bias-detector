from django.urls import path
from .views import dashboard, upload_trades

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("upload/", upload_trades, name="upload"),
]
