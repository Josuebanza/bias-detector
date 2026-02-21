from django.urls import path

from .views import dashboard, diagnostics, upload_trades

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("diagnostics/", diagnostics, name="diagnostics"),
    path("upload/", upload_trades, name="upload"),
]
