import os
import glob
import numpy as np
import soundfile as sf
import librosa
import concurrent.futures

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
INPUT_DIR = r"C:\Users\my380\Downloads\archive\VCTK-Corpus\VCTK-Corpus\wav48"
OUTPUT_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\datasets_real_audio\VCTK-Corpus"

MIN_DURATION = 1.5  # Мінімальна довжина (в секундах) ПІСЛЯ обрізки
TOP_DB = 30  # Поріг відсікання тиші (30 дБ нижче за пікову гучність вважається тишею)
MAX_WORKERS = 14  # Кількість потоків процесора

# Створюємо цільову папку, якщо її немає
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. ФУНКЦІЯ ОБРОБКИ ОДНОГО ФАЙЛУ
# ==========================================
def process_trimming(file_path):
    try:
        # Читаємо файл (soundfile працює значно швидше за librosa.load)
        y, sr = sf.read(file_path)

        # Конвертація в моно, якщо раптом файл стерео
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Якщо файл взагалі пустий або битий
        if len(y) == 0:
            return 0  # 0 означає "пропущено"

        # Обрізаємо тишу.
        # y_trimmed - це лише корисний сигнал
        # index - це масив [початок, кінець], але він нам тут не потрібен
        y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

        # Перевіряємо довжину ЗАЛИШКУ
        duration_sec = len(y_trimmed) / sr
        if duration_sec > MIN_DURATION:
            # Зберігаємо файл у загальну папку
            filename = os.path.basename(file_path)
            out_path = os.path.join(OUTPUT_DIR, filename)

            # Зберігаємо як PCM 16-bit (стандарт)
            sf.write(out_path, y_trimmed, sr, subtype='PCM_16')
            return 1  # 1 означає "успішно збережено"
        else:
            return 2  # 2 означає "занадто короткий після обрізки"

    except Exception as e:
        # Ігноруємо пошкоджені файли
        return 0


# ==========================================
# 3. ОСНОВНИЙ ПРОЦЕС
# ==========================================
if __name__ == "__main__":
    print(f"📂 Скануємо папку: {INPUT_DIR}")

    # Рекурсивний пошук ВСІХ .wav файлів у всіх підпапках (**)
    all_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.wav"), recursive=True)
    total_files = len(all_files)

    if total_files == 0:
        print("⚠️ У вказаній папці не знайдено .wav файлів! Перевірте шлях.")
    else:
        print(f"🚀 Знайдено {total_files} файлів. Починаємо обрізку тиші на {MAX_WORKERS} потоках...")

        saved_count = 0
        short_count = 0
        completed = 0
        last_percent = -1

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Відправляємо завдання в пул
            futures = {executor.submit(process_trimming, f): f for f in all_files}

            for future in concurrent.futures.as_completed(futures):
                status = future.result()

                if status == 1:
                    saved_count += 1
                elif status == 2:
                    short_count += 1

                completed += 1
                current_percent = int((completed / total_files) * 100)

                # Малюємо прогрес
                if current_percent > last_percent:
                    bar = '█' * (current_percent // 2) + '-' * (50 - (current_percent // 2))
                    print(f"\rПрогрес: [{bar}] {current_percent}% ({completed}/{total_files})", end="")
                    last_percent = current_percent

        print("\n\n🎉 ГОТОВО!")
        print(f"✅ Успішно обрізано і збережено: {saved_count} файлів")
        print(f"🗑️ Відкинуто (менше {MIN_DURATION} сек): {short_count} файлів")
        print(f"📁 Всі ідеальні файли тепер лежать у: {OUTPUT_DIR}")