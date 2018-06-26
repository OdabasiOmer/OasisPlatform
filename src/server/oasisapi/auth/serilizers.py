from django.utils.translation import ugettext_lazy as _
from rest_framework import serializers
from rest_framework_simplejwt import settings as jwt_settings
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer as BaseTokenObtainPairSerializer
from django.utils.six import text_type
from rest_framework_simplejwt.tokens import RefreshToken


class TokenObtainPairSerializer(BaseTokenObtainPairSerializer):
    def __init__(self, *args, **kwargs):
        super(TokenObtainPairSerializer, self).__init__(*args, **kwargs)

        self.fields[self.username_field].help_text = _('Your username')
        self.fields['password'].help_text = _('your password')

    def validate(self, attrs):
        data = super(TokenObtainPairSerializer, self).validate(attrs)

        refresh = self.get_token(self.user)

        data['refresh_token'] = text_type(refresh)
        data['access_token'] = text_type(refresh.access_token)
        data['token_type'] = 'Bearer'
        data['expires_in'] = jwt_settings.api_settings.REFRESH_TOKEN_LIFETIME.total_seconds()

        return data


class TokenRefreshSerializer(serializers.Serializer):
    def validate(self, attrs):
        token = self.context['request'].META['HTTP_AUTHORIZATION'][7:]  # skip 'Bearer '

        refresh = RefreshToken(token)

        data = {
            'access_token': text_type(refresh.access_token),
            'token_type': 'Bearer',
            'expires_in': jwt_settings.api_settings.ACCESS_TOKEN_LIFETIME.total_seconds(),
        }
        return data
