from django.db import models

class Feedback(models.Model):
    SATISFACTION_CHOICES = [
        ('very_satisfied', 'Very Satisfied'),
        ('satisfied', 'Satisfied'),
        ('neutral', 'Neutral'),
        ('dissatisfied', 'Dissatisfied'),
        ('very_dissatisfied', 'Very Dissatisfied'),
    ]

    ACCURACY_CHOICES = [
        ('very_accurate', 'Very Accurate'),
        ('accurate', 'Accurate'),
        ('neutral', 'Neutral'),
        ('inaccurate', 'Inaccurate'),
        ('very_inaccurate', 'Very Inaccurate'),
    ]

    name = models.CharField(max_length=100)
    email = models.EmailField()
    satisfaction = models.CharField(max_length=20, choices=SATISFACTION_CHOICES)
    accuracy = models.CharField(max_length=20, choices=ACCURACY_CHOICES)
    improvements = models.TextField()
    additional_feedback = models.TextField(blank=True)

    def __str__(self):
        return f"Feedback from {self.name}"


# predictions/models.py
from django.db import models

class LSTMParameters(models.Model):
    lstm_units = models.IntegerField(default=50)
    dropout_rate = models.FloatField(default=0.2)
    epochs = models.IntegerField(default=10)
    batch_size = models.IntegerField(default=20)

    def __str__(self):
        return f"LSTM Parameters: Units={self.lstm_units}, Dropout={self.dropout_rate}, Epochs={self.epochs}, Batch Size={self.batch_size}"

class ARIMAParameters(models.Model):
    p = models.IntegerField(default=1)
    d = models.IntegerField(default=1)
    q = models.IntegerField(default=1)

    def __str__(self):
        return f"ARIMA Parameters: p={self.p}, d={self.d}, q={self.q}"