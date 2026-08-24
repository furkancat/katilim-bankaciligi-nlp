#!/usr/bin/env python3
"""
Batch Processor — Tüm Kampanya Verisini Pipeline'dan Geçirme

Kullanım:
    python batch_processor.py --input data/raw/ --output data/structured_kampanyalar.jsonl

Girdi:  Scraper'ın ürettiği ham JSONL dosyaları (banka başına bir dosya)
Çıktı:  Pipeline'dan geçmiş yapılandırılmış JSONL (tüm bankalar bir arada)
"""

import json
import os
import glob
import argparse
from datetime import datetime
from typing import Iterator

from langgraph_pipeline import build_pipeline, PipelineState


def read_jsonl_files(input_dir: str) -> Iterator[dict]:
    """Bir dizindeki tüm .jsonl dosyalarını okur."""
    pattern = os.path.join(input_dir, "*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[UYARI] {pattern} eşleşen dosya bulunamadı!")
        return
    
    for filepath in sorted(files):
        print(f"[OKU] {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  [HATA] JSON parse: {e}")


def build_state(record: dict) -> PipelineState:
    return {
        "raw_text": record.get("detay_metin") or record.get("liste_ozet") or record.get("liste_baslik") or "",
        "banka_adi": record.get("banka_adi", "Bilinmiyor"),
        "url": record.get("liste_url", ""),
        "liste_etiket": record.get("liste_etiket", ""),
        "liste_bitis_tarihi": record.get("liste_bitis_tarihi", ""),
        "kaynak": record.get("kaynak", "bilinmiyor"),
        "regex_results": {},
        "regex_confidence": 0.0,
        "llm_results": {},
        "llm_raw_response": "",
        "validation_errors": [],
        "final_output": {},
        "retry_count": 0,
    }


def process_batch(input_dir: str, output_file: str) -> None:
    app = build_pipeline()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    total = 0
    success = 0
    failed = 0
    
    print("=" * 60)
    print("BATCH PROCESSOR BAŞLATILIYOR")
    print(f"Girdi dizini: {input_dir}")
    print(f"Çıktı dosyası: {output_file}")
    print("=" * 60)
    
    for record in read_jsonl_files(input_dir):
        total += 1
        kampanya_id = f"{record.get('kaynak', 'unknown')}_{total}"
        
        print(f"\n[{total}] İşleniyor: {kampanya_id}")
        print(f"    URL: {record.get('liste_url', 'N/A')[:60]}...")
        
        try:
            state = build_state(record)
            result = app.invoke(state)
            final = result["final_output"]
            
            output_record = {
                **final,
                "_ham_veri": {
                    "liste_baslik": record.get("liste_baslik"),
                    "liste_ozet": record.get("liste_ozet"),
                    "detay_html": record.get("detay_html"),
                }
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            
            success += 1
            print(f"    ✅ Başarılı (LLM: {final.get('llm_used')}, Conf: {final.get('regex_confidence')})")
            
        except Exception as e:
            failed += 1
            print(f"    ❌ HATA: {e}")
            error_record = {
                "_error": str(e),
                "_raw_record": record,
                "processed_at": datetime.now().isoformat(),
            }
            with open(output_file + ".errors", "a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("BATCH TAMAMLANDI")
    print(f"  Toplam:   {total}")
    print(f"  Başarılı: {success}")
    print(f"  Hatalı:   {failed}")
    print(f"  Çıktı:    {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Kampanya verisini batch işle")
    parser.add_argument("--input", default="data/raw", help="Ham JSONL dosyalarının dizini")
    parser.add_argument("--output", default="data/structured_kampanyalar.jsonl", help="Çıktı dosyası")
    args = parser.parse_args()
    
    process_batch(args.input, args.output)


if __name__ == "__main__":
    main()