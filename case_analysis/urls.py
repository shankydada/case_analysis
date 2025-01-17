from django.urls import path
from . import views
from django.contrib import admin

urlpatterns = [
    path('', views.predict, name='predict'),
    path('admin/', admin.site.urls),
]
