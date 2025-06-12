import epi_detector

if __name__ == '__main__':
    detector = epi_detector.EPIDetector(
        dataset_dir='dataset',
        img_size=128,
        batch_size=16,
        epochs=10
    )
    
    # Se ainda não treinou o modelo:
    detector.preprocess_data()
    detector.build_model()
    detector.train()
    detector.save_model()

    # Ou, se já tiver treinado o modelo e quiser apenas executar:
    # detector.load_model()

    detector.run_realtime_detection()
