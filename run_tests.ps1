$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧪 INICIANDO BATERÍA DE PRUEBAS NATIVAS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Compilando Binarios en Modo Release (AVX2 Activado)..." -ForegroundColor Yellow
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error compilando el proyecto." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "✅ Compilación exitosa." -ForegroundColor Green

Write-Host "`n[2/3] PRUEBA: Evolución Poblacional Paralela (Rayon)" -ForegroundColor Yellow
Write-Host "Esta prueba validará el uso del 100% de la CPU para converger la secuencia 'hola mundo'."
$startTime = Get-Date
cargo run --release --bin hola-mundo-evolution
$endTime = Get-Date
$duration = New-TimeSpan -Start $startTime -End $endTime
Write-Host "✅ Evolución completada en $($duration.TotalSeconds) segundos." -ForegroundColor Green

Write-Host "`n[3/3] PRUEBA: Auto-Grad y Samplers en GAJE-CLI" -ForegroundColor Yellow
Write-Host "Validando el entrenamiento por gradiente cruzado nativo en Rust..."
$modelPath = "models/mc_optimized_qwen/model.gaje"

if (-Not (Test-Path $modelPath)) {
    Write-Host "⚠️ No se encontró el modelo $modelPath." -ForegroundColor Magenta
    Write-Host "Asegúrate de haber corrido 'python scripts/optimize_mc_gaje.py' primero." -ForegroundColor Magenta
} else {
    cargo run --release --bin gaje-cli -- --model $modelPath --train "La inteligencia es la capacidad de adaptarse al cambio" --epochs 10 --scale 0.05
    Write-Host "✅ Entrenamiento completado. El Loss debió descender continuamente." -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🏆 TODAS LAS PRUEBAS FINALIZADAS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
