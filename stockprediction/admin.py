from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Feedback

admin.site.register(Feedback)


# predictions/admin.py
from django.contrib import admin
from .models import LSTMParameters, ARIMAParameters

class LSTMParametersAdmin(admin.ModelAdmin):
    list_display = ('lstm_units', 'dropout_rate', 'epochs', 'batch_size')  # Fields to display in the list view
    list_filter = ('epochs', 'batch_size')  # Fields to filter by
    search_fields = ('lstm_units',)  # Fields to search by

class ARIMAParametersAdmin(admin.ModelAdmin):
    list_display = ('p', 'd', 'q')  # Fields to display in the list view
    list_filter = ('p', 'd', 'q')  # Fields to filter by
    search_fields = ('p', 'd', 'q')  # Fields to search by


admin.site.register(LSTMParameters, LSTMParametersAdmin)
admin.site.register(ARIMAParameters, ARIMAParametersAdmin)