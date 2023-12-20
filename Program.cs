using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using OpenCvSharp;
using OpenCvSharp.Face;


class Program
{
    static void Main(string[] args)
    {

        string directorioBase = AppDomain.CurrentDomain.BaseDirectory;
        string directorioProyecto = Path.GetFullPath(Path.Combine(directorioBase, "..", ".."));

        string video1 = Path.Combine(directorioProyecto, "video1", "PruebaContador.mp4");
        string output1 = Path.Combine(directorioProyecto, "Output");

        //Si no existe la carpeta para almacenar las imagenes la creará
        CreateOutputFolder(output1);
        ProcesarVideo(video1, output1);

        string video2 = Path.Combine(directorioProyecto, "video2", "Prueba2.1.mp4"); 
        string output2 = Path.Combine(directorioProyecto, "Output2"); 

        CreateOutputFolder(output2);
        ProcesarVideoConCaras(video2, output2);

        Console.WriteLine("Proceso completo.");
    }

    //Creacion de carpeta para almacenar las imagenes
    static void CreateOutputFolder(string folderPath)
    {
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
            Console.WriteLine($"Carpeta de salida creada en: {folderPath}");
        }
        else
        {
            Console.WriteLine($"La carpeta de salida ya existe en: {folderPath}");
        }
    }


    static void ProcesarVideo(string videoPath, string outputFolderPath)
    {
        using (VideoCapture captura = new VideoCapture(videoPath))
        {
            if (!captura.IsOpened())
            {
                Console.WriteLine("Error al abrir el video.");
                return;
            }

            int frameNumero = 0;
            while (true)
            {
                Mat frame = new Mat();
                captura.Read(frame);

                if (frame.Empty())
                    break;

                string outputPath = System.IO.Path.Combine(outputFolderPath, $"frame_{frameNumero:D5}.png");
                Cv2.ImWrite(outputPath, frame);

                Console.WriteLine($"Procesando frame {frameNumero + 1}");
                frameNumero++;
            }

            Console.WriteLine("Proceso de transformación a imágenes completado.");
        }
    }

    static void ProcesarVideoConCaras(string videoPath, string outputFolderPath)
    {

        string directorioBase = AppDomain.CurrentDomain.BaseDirectory;
        string directorioProyecto = Path.GetFullPath(Path.Combine(directorioBase, "..", ".."));
        string ruteComplete = Path.Combine(directorioProyecto, "haarcascades", "haarcascade_frontalface_default.xml");

        using (VideoCapture captura = new VideoCapture(videoPath))
        using (CascadeClassifier faceCascade = new CascadeClassifier(ruteComplete))
        {
            if (!captura.IsOpened())
            {
                Console.WriteLine("Error al abrir el video.");
                return;
            }

            int frameNumero = 0;
            Mat anteriorFrame = new Mat();

            while (true)
            {
                Mat frame = new Mat();
                captura.Read(frame);

                if (frame.Empty())
                    break;

                Mat grayFrame = new Mat();
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                Rect[] caras = faceCascade.DetectMultiScale(grayFrame, 1.1, 3, HaarDetectionTypes.DoCannyPruning);

                Console.WriteLine($"Número de caras detectadas: {caras.Length}");

                if (caras.Length > 0)
                {
                    foreach (Rect cara in caras)
                    {
                        Cv2.Rectangle(frame, cara, Scalar.Red, 2);
                    }

                    // Comparar expresiones faciales en frames sucesivos
                    if (frameNumero > 0)
                    {
                        bool expresionCambiada = DetectarCambioExpresion(anteriorFrame, grayFrame, caras);
                        if (expresionCambiada)
                        {
                            // Guardar el cuadro modificado como una imagen en la carpeta de salida
                            string outputPath = Path.Combine(outputFolderPath, $"frame_{frameNumero:D5}.png");
                            Cv2.ImWrite(outputPath, frame);

                            Console.WriteLine($"Expresión cambiada en el frame {frameNumero}");
                           

                        }
                    }

                    anteriorFrame = grayFrame.Clone();

                }

                Console.WriteLine($"Procesando frame {frameNumero + 1}");
                frameNumero++;
            }

            Console.WriteLine("Proceso de análisis de caras completado.");
        }
    }

    static bool DetectarCambioExpresion(Mat frameAnterior, Mat frameActual, Rect[] caras)
    {
        Console.WriteLine("Detectando cambio de expresión..."); 

        // Verifica si hay al menos una cara en ambos frames
        if (caras.Length > 0 && frameAnterior.Size() == frameActual.Size())
        {
     
            List<Rect> regionesBocaAnterior = ObtenerRegionesBoca(frameAnterior, caras);
            List<Rect> regionesBocaActual = ObtenerRegionesBoca(frameActual, caras);

            // Compara las regiones de la boca entre frames
            for (int i = 0; i < regionesBocaAnterior.Count; i++)
            {
                Rect regionAnterior = regionesBocaAnterior[i];
                Rect regionActual = regionesBocaActual[i];

                // Compara la diferencia de intensidad promedio en las regiones de la boca
                double diferenciaIntensidad = CompararDiferenciaIntensidad(frameAnterior, frameActual, regionAnterior, regionActual);

                // Establece un umbral para determinar si hay un cambio significativo
                double umbralDiferencia = 5.0; 

                if (diferenciaIntensidad > umbralDiferencia)
                {
                    return true;
                }
            }
        }

        // No se detectaron cambios significativos en la expresión facial
        return false;
    }

    static List<Rect> ObtenerRegionesBoca(Mat frame, Rect[] caras)
    {
        List<Rect> regionesBoca = new List<Rect>();

        foreach (Rect cara in caras)
        {
            // Define la región de la boca en base a la posición de la cara
            int xBoca = cara.X + cara.Width / 4;
            int yBoca = cara.Y + 2 * cara.Height / 3;
            int anchoBoca = cara.Width / 2;
            int altoBoca = cara.Height / 3;

            // Asegura de que la región de la boca esté dentro de los límites del frame
            xBoca = Math.Max(0, xBoca);
            yBoca = Math.Max(0, yBoca);
            anchoBoca = Math.Min(anchoBoca, frame.Width - xBoca);
            altoBoca = Math.Min(altoBoca, frame.Height - yBoca);

            Rect regionBoca = new Rect(xBoca, yBoca, anchoBoca, altoBoca);
            regionesBoca.Add(regionBoca);
        }

        return regionesBoca;
    }

    static double CompararDiferenciaIntensidad(Mat frameAnterior, Mat frameActual, Rect regionAnterior, Rect regionActual)
    {
        // Extrae las regiones de interés de los frames
        Mat roiAnterior = new Mat(frameAnterior, regionAnterior);
        Mat roiActual = new Mat(frameActual, regionActual);

        // Calcula la diferencia absoluta entre las regiones
        Mat diferenciaAbsoluta = new Mat();
        Cv2.Absdiff(roiAnterior, roiActual, diferenciaAbsoluta);

        // Calcula la intensidad promedio de la diferencia
        Scalar intensidadPromedio = Cv2.Mean(diferenciaAbsoluta);

        return intensidadPromedio.Val0;
    }
}