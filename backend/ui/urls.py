from django.urls import path

from .views import dashboard, diagnostics, simulator, upload_trades

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("diagnostics/", diagnostics, name="diagnostics"),
    path("simulator/", simulator, name="simulator"),
    path("upload/", upload_trades, name="upload"),
]
