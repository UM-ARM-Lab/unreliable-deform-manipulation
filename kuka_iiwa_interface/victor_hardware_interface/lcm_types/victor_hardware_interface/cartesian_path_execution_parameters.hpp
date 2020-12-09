/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by lcm-gen
 **/

#include <lcm/lcm_coretypes.h>

#ifndef __victor_hardware_interface_cartesian_path_execution_parameters_hpp__
#define __victor_hardware_interface_cartesian_path_execution_parameters_hpp__

#include "victor_hardware_interface/cartesian_value_quantity.hpp"
#include "victor_hardware_interface/cartesian_value_quantity.hpp"

namespace victor_hardware_interface
{

class cartesian_path_execution_parameters
{
    public:
        victor_hardware_interface::cartesian_value_quantity max_velocity;

        victor_hardware_interface::cartesian_value_quantity max_acceleration;

        double     max_nullspace_velocity;

        double     max_nullspace_acceleration;

    public:
        /**
         * Encode a message into binary form.
         *
         * @param buf The output buffer.
         * @param offset Encoding starts at thie byte offset into @p buf.
         * @param maxlen Maximum number of bytes to write.  This should generally be
         *  equal to getEncodedSize().
         * @return The number of bytes encoded, or <0 on error.
         */
        inline int encode(void *buf, int offset, int maxlen) const;

        /**
         * Check how many bytes are required to encode this message.
         */
        inline int getEncodedSize() const;

        /**
         * Decode a message from binary form into this instance.
         *
         * @param buf The buffer containing the encoded message.
         * @param offset The byte offset into @p buf where the encoded message starts.
         * @param maxlen The maximum number of bytes to reqad while decoding.
         * @return The number of bytes decoded, or <0 if an error occured.
         */
        inline int decode(const void *buf, int offset, int maxlen);

        /**
         * Retrieve the 64-bit fingerprint identifying the structure of the message.
         * Note that the fingerprint is the same for all instances of the same
         * message type, and is a fingerprint on the message type definition, not on
         * the message contents.
         */
        inline static int64_t getHash();

        /**
         * Returns "cartesian_path_execution_parameters"
         */
        inline static const char* getTypeName();

        // LCM support functions. Users should not call these
        inline int _encodeNoHash(void *buf, int offset, int maxlen) const;
        inline int _getEncodedSizeNoHash() const;
        inline int _decodeNoHash(const void *buf, int offset, int maxlen);
        inline static uint64_t _computeHash(const __lcm_hash_ptr *p);
};

int cartesian_path_execution_parameters::encode(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;
    int64_t hash = (int64_t)getHash();

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int cartesian_path_execution_parameters::decode(const void *buf, int offset, int maxlen)
{
    int pos = 0, thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int cartesian_path_execution_parameters::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t cartesian_path_execution_parameters::getHash()
{
    static int64_t hash = _computeHash(NULL);
    return hash;
}

const char* cartesian_path_execution_parameters::getTypeName()
{
    return "cartesian_path_execution_parameters";
}

int cartesian_path_execution_parameters::_encodeNoHash(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;

    tlen = this->max_velocity._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->max_acceleration._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __double_encode_array(buf, offset + pos, maxlen - pos, &this->max_nullspace_velocity, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __double_encode_array(buf, offset + pos, maxlen - pos, &this->max_nullspace_acceleration, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int cartesian_path_execution_parameters::_decodeNoHash(const void *buf, int offset, int maxlen)
{
    int pos = 0, tlen;

    tlen = this->max_velocity._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->max_acceleration._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __double_decode_array(buf, offset + pos, maxlen - pos, &this->max_nullspace_velocity, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __double_decode_array(buf, offset + pos, maxlen - pos, &this->max_nullspace_acceleration, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int cartesian_path_execution_parameters::_getEncodedSizeNoHash() const
{
    int enc_size = 0;
    enc_size += this->max_velocity._getEncodedSizeNoHash();
    enc_size += this->max_acceleration._getEncodedSizeNoHash();
    enc_size += __double_encoded_array_size(NULL, 1);
    enc_size += __double_encoded_array_size(NULL, 1);
    return enc_size;
}

uint64_t cartesian_path_execution_parameters::_computeHash(const __lcm_hash_ptr *p)
{
    const __lcm_hash_ptr *fp;
    for(fp = p; fp != NULL; fp = fp->parent)
        if(fp->v == cartesian_path_execution_parameters::getHash)
            return 0;
    const __lcm_hash_ptr cp = { p, (void*)cartesian_path_execution_parameters::getHash };

    uint64_t hash = 0xa5c0017a3cb11d5dLL +
         victor_hardware_interface::cartesian_value_quantity::_computeHash(&cp) +
         victor_hardware_interface::cartesian_value_quantity::_computeHash(&cp);

    return (hash<<1) + ((hash>>63)&1);
}

}

#endif
