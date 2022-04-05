#include "StreamScatter.h"
#include "math.h"

StreamScatter::StreamScatter(int dataWidth, int dataHeight, int numStreams)
    : m_dataSize(make_int2(dataWidth, dataHeight)),
      m_numStreams(numStreams)
{
}

uint2 StreamScatter::start(int stream) const
{
    uint2 ret;
    ret.x = 0;
    ret.y = stream * ceil_div(m_dataSize.y, m_numStreams);
    return ret;
}

uint2 StreamScatter::end(int stream) const
{
    uint2 ret;
    ret.x = m_dataSize.x;
    ret.y = min(m_dataSize.y, (stream + 1) * ceil_div(m_dataSize.y, m_numStreams));
    return ret;
}

uint2 StreamScatter::size(int stream) const
{
    const uint2 streamStart = start(stream);
    const uint2 streamEnd = end(stream);
    return streamEnd - streamStart;
}