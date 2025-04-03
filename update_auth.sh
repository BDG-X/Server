#!/bin/bash
# Replace all API key authentication requirements with "No Authentication Required"
sed -i 's/<strong>Authentication Required:<\/strong> Header <code>X-API-Key<\/code> must be provided./<strong>No Authentication Required:<\/strong> This endpoint is publicly accessible./g' app.py
sed -i 's/<strong>Authentication Required:<\/strong> Header <code>X-Admin-Key<\/code> must be provided./<strong>No Authentication Required:<\/strong> This endpoint is publicly accessible./g' app.py
