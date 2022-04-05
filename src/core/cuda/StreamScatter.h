#pragma once

#include <cuda_runtime.h>

/**
 * Classe permettant de simplifier le partage du travail entre les streams.
 *
 * Fournit, en 2D, suivant une taille de données, et un nombre de streams,
 * les limites de chaque stream.
 *
 * On subdivise ici par lignes car on considère que une même ligne est toujours contigue
 * (exploitation de la data locality).
 */
class StreamScatter
{
public:
    StreamScatter(int dataWidth, int dataHeight, int numStreams);

    /**
     * Récupère les indices "start" pour ce stream
     */
    uint2 start(int stream) const;

    /**
     * Récupèr les indices "end" pour ce stream
     */
    uint2 end(int stream) const;

    /**
     * Récupère la taille pour ce stream
     */
    uint2 size(int stream) const;

private:
    int2 m_dataSize;
    int m_numStreams;    
};