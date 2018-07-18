# -*- coding: utf-8 -*-
# Generated by Django 1.11.13 on 2018-07-17 15:48
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import model_utils.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='AnalysisModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', model_utils.fields.AutoCreatedField(default=django.utils.timezone.now, editable=False, verbose_name='created')),
                ('modified', model_utils.fields.AutoLastModifiedField(default=django.utils.timezone.now, editable=False, verbose_name='modified')),
                ('supplier_id', models.CharField(help_text='The supplier ID for the model.', max_length=255)),
                ('model_id', models.CharField(help_text='The model ID for the model.', max_length=255)),
                ('version_id', models.CharField(help_text='The version ID for the model.', max_length=255)),
                ('keys_server_uri', models.URLField(help_text='The root url for the model server.')),
                ('creator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='analysismodel',
            unique_together=set([('supplier_id', 'model_id', 'version_id')]),
        ),
    ]
