from django import template

register = template.Library()

@register.filter
def safedict(value, key):
    return value.get(key, 'No description available.') if isinstance(value, dict) else 'No description available.'