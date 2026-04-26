# pontia-mlops-tutorial-said-elkhababi

## Integrantes
- Tu Nombre y Apellido

## Descripción del proyecto
Este proyecto implementa una pipeline básica de MLOps/DevOps para entrenar, validar, versionar y desplegar un modelo de Machine Learning usando GitHub Actions y Render.

## Estructura del repositorio
- `.github/workflows/`: workflows de integración, build y deploy
- `src/`: código de entrenamiento, preprocesado y evaluación
- `unit_tests/`: tests unitarios
- `model_tests/`: tests del modelo entrenado
- `deployment/`: API en FastAPI para servir predicciones en Render
- `requirements.txt`: dependencias para desarrollo, tests y build
- `deployment/requirements.txt`: dependencias para despliegue de la API

## Pipelines
### Integration
Ejecuta tests unitarios en cada Pull Request para validar cambios antes del merge a `main`.

### Build
Se ejecuta al hacer push a `main`. Descarga el dataset, entrena el modelo, guarda los artefactos (`model.pkl`, `scaler.pkl`, `encoders.pkl`) y crea automáticamente una release del modelo en GitHub.

### Deploy
Permite lanzar manualmente el despliegue hacia Render usando un deploy hook.

## Despliegue
La API está desplegada en Render y carga automáticamente los artefactos del último release publicado en GitHub.

## Endpoints de la API
- `GET /health`
- `POST /predict`
- `GET /metrics`

## Variables y secretos usados
### Variables de entorno en Render
- `GITHUB_REPO`
- `GITHUB_TOKEN`

### Secretos en GitHub
- `RENDER_DEPLOY_HOOK`

## Flujo de trabajo con Git
Los cambios se realizan en ramas separadas, se abren Pull Requests hacia `main` y se validan mediante checks automáticos antes del merge.

## Rollback

El rollback permite volver a una versión anterior estable si una nueva versión del modelo o del servicio falla después del despliegue.

En este proyecto, el rollback puede realizarse restaurando una release anterior del modelo en GitHub Releases y ejecutando de nuevo el despliegue en Render.

Pasos generales:

1. Identificar la última release estable del modelo en GitHub Releases.
2. Restaurar o seleccionar los artefactos de esa release anterior.
3. Ejecutar de nuevo el workflow de despliegue `deploy.yml`.
4. Verificar que el servicio en Render responde correctamente.
5. Probar los endpoints principales: `/health`, `/metrics` y `/predict`.

Este proceso permite recuperar una versión funcional del servicio sin rehacer todo el pipeline desde cero.

## Evidencia
La API puede probarse desde `/docs`, Postman o `curl`.

## Estado actual
- Integración continua configurada
- Build automático configurado
- Release automática del modelo configurada
- Despliegue en Render operativo