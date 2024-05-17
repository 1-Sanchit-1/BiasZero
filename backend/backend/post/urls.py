from django.urls import path
from . import views

urlpatterns = [
    path('save-qa/', views.QuestionAnswerAPIView.as_view()),
    # path('posts/', views.QNAUploadView.as_view(), name= 'QNAUploadView'),
    path('',views.home,name='home')
]
