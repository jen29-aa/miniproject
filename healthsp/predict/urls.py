from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.home,name='home'),
      path('predict/',views.predict,name='predict'),
        path('contact/',views.contact,name='contact'),
            path('about/',views.about,name='about'),
              path('article/',views.article,name='article'),
             

]