from django.urls import path
from stockprediction import views


urlpatterns = [
path('',views.index,name='index'),
path('search',views.search,name='search'),
path('stock/<str:pk>',views.stock,name='stock'),
path('get/stock', views.loadstock, name = "loadstock"),
path('login',views.loginPage,name='login'),
path('logout',views.logoutUser,name='logout'),
path('register',views.registerPage,name='register'),
path('feedback/', views.feedback_view, name='feedback_form'),
path('predict/', views.predict_stock, name='predict_stock'),
path('fundamental/', views.get_fundamental_data, name='fundamental'),


    
]
