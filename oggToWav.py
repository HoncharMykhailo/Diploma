import os
import numpy as np
import soundfile as sf
import concurrent.futures

# ==========================================
# НАЛАШТУВАННЯ ШЛЯХІВ
# ==========================================
INPUT_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\tg downloads_real"
OUTPUT_DIR = r"/datasets_real_audio/tg_downloads"

MAX_WORKERS = 14
CHUNK_DURATION = 2.0  # Довжина шматка в секундах

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_file(filename):
    in_path = os.path.join(INPUT_DIR, filename)
    base_name = filename.replace('.ogg', '')

    try:
        # Читаємо оригінальний файл
        data, sr = sf.read(in_path)

        # Перетворюємо в моно, якщо файл стерео (librosa робить це за замовчуванням)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        samples_per_chunk = int(sr * CHUNK_DURATION)
        min_samples = samples_per_chunk // 2

        chunk_count = 0

        # Нарізаємо аудіо
        for i in range(0, len(data), samples_per_chunk):
            chunk = data[i: i + samples_per_chunk]

            # Відкидаємо шматок, якщо він коротший за 1 секунду
            if len(chunk) < min_samples:
                continue

            # Доповнюємо нулями (тишею), якщо шматок від 1 до 1.99 секунд
            if len(chunk) < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)), mode='constant')

            # Зберігаємо шматок: ім'я_файлу_chunk001.wav
            out_filename = f"{base_name}_chunk{chunk_count:03d}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            sf.write(out_path, chunk, sr, subtype='PCM_16')
            chunk_count += 1

        return True, chunk_count
    except Exception as e:
        print(f"\n⚠️ Помилка з файлом {filename}: {e}")
        return False, 0


if __name__ == "__main__":
    print(f"📂 Відкриваємо папку: {INPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.ogg')]
    total_files = len(files)

    if total_files == 0:
        print("⚠️ У вказаній папці не знайдено .ogg файлів!")
    else:
        print(f"🚀 Починаємо паралельну нарізку та конвертацію {total_files} файлів...")

        total_chunks_created = 0
        success_count = 0
        completed = 0
        last_percent = -1

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file, f): f for f in files}

            for future in concurrent.futures.as_completed(futures):
                success, chunks = future.result()
                if success:
                    success_count += 1
                    total_chunks_created += chunks

                completed += 1
                current_percent = int((completed / total_files) * 100)

                if current_percent > last_percent:
                    bar = '█' * (current_percent // 2) + '-' * (50 - (current_percent // 2))
                    print(f"\rПрогрес: [{bar}] {current_percent}% ({completed}/{total_files})", end="")
                    last_percent = current_percent

        print(f"\n🎉 Готово! Успішно оброблено {success_count} з {total_files} файлів.")
        print(f"🔪 Загалом створено 2-секундних фрагментів: {total_chunks_created}")