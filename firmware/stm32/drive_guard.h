#ifndef ROBOPACK_DRIVE_GUARD_H
#define ROBOPACK_DRIVE_GUARD_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define RP_PACKET_SIZE 16
#define RP_MAX_SPEED_MM_S 600
#define RP_MAX_STEERING_MRAD 600
#define RP_MAX_LEASE_MS 250
/* 'R','P', version=1, enable, seq:u32, speed:u16, steering:i16,
 * lease:u16, crc16:u16. Little endian, CRC-16/CCITT-FALSE over first 14 bytes. */
typedef struct {
    uint32_t sequence;
    uint16_t speed_mm_s, lease_ms;
    int16_t steering_mrad;
    uint8_t enable;
} rp_command;
typedef struct {
    uint8_t armed, estop_latched, have_sequence, enabled;
    uint32_t last_sequence, received_ms, last_byte_ms;
    uint16_t lease_ms, speed_mm_s;
    int16_t steering_mrad;
    uint8_t buffer[RP_PACKET_SIZE];
    size_t used;
} rp_guard;
uint16_t rp_crc16(const uint8_t* data,size_t length);
void rp_encode(const rp_command* command,uint8_t packet[RP_PACKET_SIZE]);
int rp_decode(const uint8_t packet[RP_PACKET_SIZE],rp_command* command);
void rp_guard_init(rp_guard* guard);
/* Board-local operator action ONLY. physical_estop comes from a GPIO. */
int rp_guard_arm(rp_guard* guard,int physical_estop);
void rp_guard_tick(rp_guard* guard,uint32_t now_ms,int physical_estop);
void rp_guard_feed(rp_guard* guard,const uint8_t* bytes,size_t length,uint32_t now_ms);
#ifdef __cplusplus
}
#endif
#endif
