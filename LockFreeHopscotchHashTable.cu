//#include"cutil.h"			// Comment this if cutil.h is not available
#include"cuda_runtime.h"
#include"stdio.h"

// Number of operations
//#define NUM_ITEMS 50000
// Number operations per block
#define FACTOR 1
// Number of integer keys assumed in the range [10, 9+KEYS]
//#define KEYS 100
// Number of threads per block
#define THREADS_NUM 32
// Number of hash table buckets
#define BUCKETS_NUM 36419

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

#if __WORDSIZE == 64
typedef unsigned long long int LL;
#else
typedef unsigned int LL;
#endif 

// Definition of generic slot
typedef LL Slot; 

#if __WORDSIZE == 64
	// Size of the neighborhood, every bucket has (1 + NEIGHBORHOOD_SIZE) slots
	#define NEIGHBORHOOD_SIZE 31

	// Because of the flag MASK, the key value in the Slot need to be restricted
	#define MAX_KEY ((LL)0x000000000fffffff)

	// Use MASK to get the flag value in Slot
	#define EMP_FLAG_MASK ((LL)0x8000000000000000)
	#define CHECK_1_FLAG_MASK ((LL)0x4000000000000000)
	#define CHECK_2_FLAG_MASK ((LL)0x2000000000000000)
	#define SWAP_FLAG_MASK ((LL)0x1000000000000000)

	#define BITMAP_MASK ((LL)0x0ffffffff0000000)
	#define BITMAP_SHIFT 28

	#define WRONG_POS ((LL)0xffffffffffffffff)

#else 
	#define NEIGHBORHOOD_SIZE 15

	#define MAX_KEY ((LL)0x00000fff)

	#define EMP_FLAG_MASK ((LL)0x80000000)
	#define CHECK_1_FLAG_MASK ((LL)0x40000000)
	#define CHECK_2_FLAG_MASK ((LL)0x20000000)
	#define SWAP_FLAG_MASK ((LL)0x10000000)

	#define BITMAP_MASK ((LL)0x0ffff000)
	#define BITMAP_SHIFT 12

	#define WRONG_POS ((LL)0xffffffff)

#endif 

#define BIT ((LL)0x1)

#define BUCKET_RANGE (NEIGHBORHOOD_SIZE+1)
// Actuall hash table pysical size
#define TABLE_SIZE (BUCKETS_NUM + NEIGHBORHOOD_SIZE)
#define MAX_PROBES_FOR_EMPTY_BUCKET (12*BUCKET_RANGE)

__device__ Slot * m_slots;			// Array of hash table slots

// Kernel for initializing device memory
// This kernel initializes every slot as an empty node
__global__ void init(Slot * slots)
{       
  	m_slots = slots;
}

// Hash function
__device__ int Hash(LL x)
{
  	return x%BUCKETS_NUM;
}

__device__ bool CompareAndSet(int pos, LL old_value, LL new_value)
{	
	Slot old_value_out = atomicCAS(&(m_slots[pos]), old_value, new_value);
	if (old_value_out == old_value) return true;
	return false;
}

__device__ void Find(LL key, LL * result, Slot * location)
{
	int tid = threadIdx.x;
	int pos = Hash(key); // step 0
	LL bitmap;
	Slot location_pos;

	do{
		*location = m_slots[pos + tid];
		location_pos = __shfl(*location, 0);
	} while( (location_pos & CHECK_1_FLAG_MASK) != 0 ); // step 2a

	// step 2b
	if( (location_pos & EMP_FLAG_MASK) != 0 ){ // step 2b1
		bitmap = (BITMAP_MASK >> BITMAP_SHIFT);
	} else { // step 2b2
		bitmap = ( (location_pos & BITMAP_MASK) >> BITMAP_SHIFT);
	}

	int predict = 0;
	int tmp_pos = Hash((*location) & MAX_KEY);

	if( (((bitmap >> tid) & BIT) != 0) // is valid
		&& ( ( (*location) & EMP_FLAG_MASK) == 0) // no emp flag
		&& ( ( (*location) & MAX_KEY) == key ) // is the key
		&& (tmp_pos == pos) // just for safe
	){
		predict = 1;
	}

	int ans = __ffs(__ballot(predict));

	if(ans==0){
		*result = WRONG_POS;
	} else {
		*result = pos + (ans - 1);
	}
}

__device__ void Delete(LL key, LL * result)
{
	int tid = threadIdx.x;
	int pos = Hash(key); // step 0
	LL target;
	Slot location;
	Slot location_pos;
	Slot new_location_pos;
	LL ans;
	bool success;

	while (true) {
		ans = WRONG_POS;
		target = WRONG_POS;
		success = false;

		Find(key, &target, &location); // step 1

		if(target == WRONG_POS){
			*result = 0; //return false
			return; //step 2b
		}

		location_pos = __shfl(location, 0);

		if( ((location_pos & CHECK_1_FLAG_MASK) != 0)
		|| ((location_pos & CHECK_2_FLAG_MASK) != 0)
		|| ((location_pos & SWAP_FLAG_MASK) != 0) ){
			;

		} else if( ((location_pos & EMP_FLAG_MASK) == 0) 
				&& ( ( ( (location_pos & BITMAP_MASK) >> BITMAP_SHIFT) & BIT ) != 0 )
				&& ( ( location_pos & MAX_KEY) == key ) ) {

			if(tid == 0){
				new_location_pos = (location_pos | EMP_FLAG_MASK);

				success = CompareAndSet(pos, location_pos, new_location_pos);
      			if (success) {
      				ans = 1; // return true;
      			} 
			}
			
			ans = __shfl(ans, 0);
			if(ans == 1){
				*result = 1;
				return;
			}

		} else {
			new_location_pos = (location_pos | CHECK_1_FLAG_MASK);

			if(tid == 0){
				/*
				if(pos == 7468){
					printf("Delete key: %lu, step 3c add CHECK_1_FLAG_MASK\n", key);
				}*/

				success = CompareAndSet(pos, location_pos, new_location_pos);
			}
				
			success = __shfl(success, 0);
			if(success){
				location_pos = new_location_pos;
				      		
      			int lane_id = (int)target - pos;
      			
      			if(tid == lane_id){ 
      				Slot new_location = (location | EMP_FLAG_MASK);
      				success = CompareAndSet(target, location, new_location); 

      				if(success){ // step 4a
      					new_location_pos = (location_pos  & (~CHECK_1_FLAG_MASK));
      					//remove bitmap bit
      					new_location_pos &= (~(BIT<<(BITMAP_SHIFT+lane_id)));

      					success = CompareAndSet(pos, location_pos, new_location_pos); 
      					if(success){
      						ans = 1;
      					} else {
      						// TODO: design fail
      						printf("Delete key: %lu, step4a2 design fail\n", key);
      					}
      				
      				} else { //step 4b
      					new_location_pos = (location_pos & (~CHECK_1_FLAG_MASK));

      					success = CompareAndSet(pos, location_pos, new_location_pos); 
      				
      					if (!success) {
      						// TODO: design fail
      						printf("Delete key: %lu, step4b2 design fail\n", key);
      					} 
      				}

      			}

      			ans = __shfl(ans, lane_id);
				if(ans == 1){
					*result = 1;
					return;
				}

      		} // else step 3c2

		}

	} 

}

__device__ void Insert(LL key, LL * result)
{
	int tid = threadIdx.x;
	int pos = Hash(key); // step 0
	LL target;
	Slot location;
	Slot location_pos;
	Slot new_location_pos;
	LL ans;
	bool success;
	//Slot location_swap_empty;
	Slot location_swap;
	Slot location_check2;
	int search_pos = pos;

	while (true) {
		ans = WRONG_POS;
		target = WRONG_POS;
		success = false;

		Find(key, &target, &location); // step 1

		if(target != WRONG_POS){
			*result = 0; // return false
			return; // step 2b
		}

		location_pos = __shfl(location, 0);

		// step 3
		if( ((location_pos & CHECK_1_FLAG_MASK) != 0)
		|| ((location_pos & CHECK_2_FLAG_MASK) != 0)
		|| ((location_pos & SWAP_FLAG_MASK) != 0) ){ // step 3a
			continue;
		} else if( (location_pos & EMP_FLAG_MASK) != 0  ){ // step 3b

			if(tid == 0){
				new_location_pos = (key & MAX_KEY);
				// add bitmap bit;
				new_location_pos |= (BIT<<(BITMAP_SHIFT));
				success = CompareAndSet(pos, location_pos, new_location_pos);

				if(success){
					ans = 1;
				}
			}
			
			ans = __shfl(ans, 0);
			if(ans == 1){ // step 3b1
				*result = 1;
				return;
			} else { // step 3b2
				continue;
			}

		} else { // step 3c
			bool continue_3c = false;

			if(tid == 0){
				new_location_pos = (location_pos | CHECK_1_FLAG_MASK);
				/*
				if(pos == 7468){
					printf("Insert key: %lu, step 3c add CHECK_1_FLAG_MASK\n", key);
					printf("location_pos: %x%x, new_location_pos: %x%x\n", location_pos, new_location_pos);
				}*/

				success = CompareAndSet(pos, location_pos, new_location_pos);

				if(!success){ // step 3c2		
					continue_3c = true;
				} else {
					location = new_location_pos;
				}
			}

			continue_3c = __shfl(continue_3c, 0);
			if(continue_3c) continue;

			location_pos = __shfl(location, 0);			
		}

		search_pos = pos;
step_4:
		//__syncthreads();

		bool condition_4b = (((location & CHECK_1_FLAG_MASK) != 0) && ((location & EMP_FLAG_MASK) != 0));
		LL target_4b = __ffs(__ballot(condition_4b));

		bool condition_4a = (((location & CHECK_1_FLAG_MASK) == 0) && ((location & EMP_FLAG_MASK) != 0));
		LL target_4a_list = __ballot(condition_4a);

		for(int target_4a_offset =  __ffs(target_4a_list); 
				target_4a_offset != 0; 
				target_4a_offset = __ffs(target_4a_list) ){

			LL lanid_4a = target_4a_offset-1;

			target = search_pos + lanid_4a;
			
			bool goto_4a_step7 = false;
			if(tid == lanid_4a){
				Slot new_location = (location | CHECK_1_FLAG_MASK);
				/*
				if(target == 7468){
					printf(" Delete key: %lu, step 4a add CHECK_1_FLAG_MASK\n", key);
				}*/

				success = CompareAndSet(target, location, new_location);
				if(success){
					location = new_location;
					//location_swap_empty = location;
					goto_4a_step7 = true;
				}
			}

			goto_4a_step7 = __shfl(goto_4a_step7, lanid_4a);

			if(goto_4a_step7){
				//target = __shfl(target, lanid_4a);
				//location_swap_empty = __shfl(location_swap_empty, lanid_4a); // use for swap
				goto step_7;  
			}

			// bug fixed: should be target_4a_list &= (~(BIT<<lanid_4a));
			// target_4a_list &= (~(BIT<<target_4a_list));
			target_4a_list &= (~(BIT<<lanid_4a));
		}

		if(target_4b != 0){
			search_pos = pos;
			goto step_6;
		}

		// step 5
		search_pos += BUCKET_RANGE;

		if( search_pos >= pos + MAX_PROBES_FOR_EMPTY_BUCKET || search_pos >= BUCKETS_NUM ){
			bool goto_5a_full = false;

			if(tid == 0){
				new_location_pos = (location_pos & (~CHECK_1_FLAG_MASK));

				success = CompareAndSet(pos, location_pos, new_location_pos);
				if(success){ // step 5a1
					goto_5a_full = true;
				} else { // step 5a2
					// TODO: design fail
					printf("Insert key: %lu, step5a2 design fail\n", key);
				}

			}
			
			goto_5a_full = __shfl(goto_5a_full, 0);
			if(goto_5a_full){
				// TODO: full
				return;
			} else {
				// TODO: design fail 5a2
				printf("Insert key: %lu, step5a2 full design fail\n", key);
			}
		}

step_6:
		location = m_slots[search_pos + tid];
		goto step_4;

step_7:
		if( ((int)target > pos) && (((int)target - NEIGHBORHOOD_SIZE) <= pos) ){ // step 7a
			location = m_slots[pos + tid];
			location_pos = __shfl(location,0);

			int lanid_7a = (int)target - pos;

			if(tid == lanid_7a){
				Slot new_location_target = (key & MAX_KEY);
				success = CompareAndSet(target, location, new_location_target);
				if(success){
					new_location_pos = (location_pos & (~CHECK_1_FLAG_MASK));
					//add bitmap bit
					new_location_pos |= (BIT<<(BITMAP_SHIFT+lanid_7a));
					/*
					if(pos == 7468){
						printf("Insert key: %lu, step 7a remove CHECK_1_FLAG_MASK\n", key);
						printf("location_pos: %x%x, new_location_pos: %x%x\n", location_pos, new_location_pos);
					}*/

					success = CompareAndSet(pos, location_pos, new_location_pos);
					if(success){
						ans = 1;
					} else {
						// TODO: design fail
						printf("Insert key: %lu, step7a1b design fail\n", key);
					}

				} else {
					new_location_pos = (location_pos & (~CHECK_1_FLAG_MASK));
					success = CompareAndSet(pos, location_pos, new_location_pos);
					if( !success ){
						// TODO: design fail
						printf("Insert key: %lu, step7a2b design fail\n", key);
					}
				}
			}

			ans = __shfl(ans, lanid_7a);
			if(ans == 1){ // step 3a1a
				*result = 1;
				return;
			} else {
				continue; 
			}
		}

		//step 8
		int to_check_2 = target - NEIGHBORHOOD_SIZE;
		location = m_slots[to_check_2 + tid];
		location_check2 = __shfl(location, 0);
step_9:
		if( ((location_check2 & CHECK_1_FLAG_MASK) == 0)
			&& ((location_check2 & CHECK_2_FLAG_MASK) == 0)
			&& ((location_check2 & SWAP_FLAG_MASK) == 0) 
			&& ((location_check2 & EMP_FLAG_MASK) == 0) ){ // step 9a

			bool goto_9a1_step12 = false;
			if(tid == 0){
				Slot new_location_check2 = (location_check2 | CHECK_2_FLAG_MASK);
				new_location_check2 |= (BIT<<(BITMAP_SHIFT+(target-to_check_2)));

				success = CompareAndSet(to_check_2, location_check2, new_location_check2);
				if(success){
					location = new_location_check2;
					location_check2 = location;
					goto_9a1_step12 = true;
				}
			} 

			goto_9a1_step12 = __shfl(goto_9a1_step12, 0);
			if(goto_9a1_step12){
				location_check2 = __shfl(location_check2, 0);
				goto step_12;
			}

		} else if( ((location_check2 & CHECK_1_FLAG_MASK) == 0)
			&& ((location_check2 & EMP_FLAG_MASK) != 0) ) { // step 9b
			// bug fixed: add CHECK_1_FLAG_MASK to location_check2 and remove target’s CHECK_1_FLAG_MASK

			int lanid_9b = (int)target - to_check_2;
			bool goto_9b_step7 = false;

			if(tid == lanid_9b){
				Slot new_location_check2 = (location_check2 | CHECK_1_FLAG_MASK);

				success = CompareAndSet(to_check_2, location_check2, new_location_check2);
				if(success){
					location_check2 = new_location_check2;

					Slot new_location_target = (location & (~CHECK_1_FLAG_MASK));

					success = CompareAndSet(target, location, new_location_target);
					if(success){
						goto_9b_step7 = true;
					} else {
						// TODO : design fail
					}
				} else {
					// TODO : design fail
				}
			}
			
			location_check2 = __shfl(location_check2,lanid_9b);
			goto_9b_step7 = __shfl(goto_9b_step7,lanid_9b);

			if(goto_9b_step7){
				target = to_check_2;
				goto step_7;
			}
			
		} else if( ((location_check2 & CHECK_1_FLAG_MASK) != 0)
			&& ((location_check2 & EMP_FLAG_MASK) != 0) ) { // step 9c
			// bug fixed: remove target’s CHECK_1_FLAG_MASK and change search_pos = to_check_2

			int lanid_9c = (int)target - to_check_2;
			bool goto_9c_step6 = false;

			if(tid == lanid_9c){
				Slot new_location_target = (location & (~CHECK_1_FLAG_MASK));

				success = CompareAndSet(target, location, new_location_target);
				if(success){
					goto_9c_step6 = true;
				} else {
					// TODO : design fail
				}
			}
			
			goto_9c_step6 = __shfl(goto_9c_step6,lanid_9c);

			if(goto_9c_step6){
				search_pos = to_check_2;
				goto step_6;
			}
		}

step_10:
		to_check_2++;
		location = m_slots[to_check_2 + tid];

		//step 11
		if(to_check_2 < (int)target){
			location_check2 = __shfl(location, 0);
			goto step_9;
		} else { // to_check_2 == target

			bool goto_11b1_full = false;

			if(tid == 0){
				Slot new_location = (location & (~CHECK_1_FLAG_MASK));
				success = CompareAndSet(target, location, new_location);
				if(success){
					location = new_location;
					
					new_location_pos = (location_pos & (~CHECK_1_FLAG_MASK));
					success = CompareAndSet(pos, location_pos, new_location_pos);
					if(success){ // step 11b1a
						goto_11b1_full = true;
					} else { // step 11b1b
						// TODO: design fail
						printf("Insert key: %lu, step11b1b design fail\n", key);
					}

				} else { // step 11b2
					// TODO: design fail
					printf("Insert key: %lu, step11b2 design fail\n", key);
				}
			}
			
			goto_11b1_full = __shfl(goto_11b1_full, 0);
			if(goto_11b1_full){
				// TODO: full
				return;

			} else {
				// TODO: design fail 11b2
				printf("Insert key: %lu, step11b2 full design fail\n", key);
			}

		}

step_12:
		if ( (location_check2 & (BIT<<BITMAP_SHIFT)) != 0 ){ // step 12a;

			int lanid_12a = target - to_check_2;

			bool goto_12a1a_step7 = false;

			if(tid == lanid_12a){
				Slot new_location = (location & (~MAX_KEY)) | (location_check2 & MAX_KEY);
				new_location &= (~EMP_FLAG_MASK);
				new_location &= (~CHECK_1_FLAG_MASK);
				new_location &= (~BITMAP_MASK);

				success = CompareAndSet(target, location, new_location);

				if(success){ // step 12a1
					location = new_location;

					Slot new_location_check2 = (location_check2 & (~CHECK_2_FLAG_MASK)) | EMP_FLAG_MASK;
					success = CompareAndSet(to_check_2, location_check2, new_location_check2);

					if(success){ // step 12a1a
						location_check2 = new_location_check2;
						goto_12a1a_step7 = true;
					} else { // step 12a1b
						// TODO: design fail
						printf("Insert key: %lu, step12a1b design fail\n", key);
					}

				} else { // step 12a2
					// TODO: design fail
					printf("Insert key: %lu, step12a2 design fail\n", key);
				}
			}

			goto_12a1a_step7 = __shfl(goto_12a1a_step7, lanid_12a);
			if(goto_12a1a_step7){
				location_check2 = __shfl(location_check2, lanid_12a);
				// bug fixed: target = to_check_2
				target = to_check_2;
				goto step_7;
			}

		} 

		// step 12b   

		LL bitmap = ( (location_check2 & BITMAP_MASK) >> BITMAP_SHIFT);
		int predict = 0; 

		if( (((bitmap >> tid) & BIT) != 0) // is valid
			&& ((location & CHECK_1_FLAG_MASK) == 0)
			&& ((location & CHECK_2_FLAG_MASK) == 0)
			&& ((location & SWAP_FLAG_MASK) == 0) ){
			predict = 1;
		}

		LL swap_list = __ballot(predict);

		// step 13
		for(int to_swap_offset =  __ffs(swap_list); 
				to_swap_offset != 0 && (to_swap_offset-1) < (int)target-to_check_2 ; 
				to_swap_offset = __ffs(swap_list) ){

			to_swap_offset--;

			// step 14
			int to_swap = to_check_2 + to_swap_offset;
			location_swap = __shfl(location, to_swap_offset);
			int lanid_target = target-to_check_2;

			// TODO: lanid == to_swap_offset 's location need to change?

			if( (location_swap & EMP_FLAG_MASK) != 0 ){ // step 14a

				bool goto_14a1a1_step7 = false;
				bool goto_14a1a2_step6 = false;

				// bug fixed: first put CHECK_1_FLAG_MASK on location_swap 

				if(tid == lanid_target){

					Slot new_location_swap = location_swap | CHECK_1_FLAG_MASK;
					success = CompareAndSet(to_swap, location_swap, new_location_swap);
					if(success){ // step 14a1
						//location_swap = new_location_swap;

						Slot new_location_check2 = (location_check2 & (~CHECK_2_FLAG_MASK));
						new_location_check2 &= (~(BIT<<(BITMAP_SHIFT+(lanid_target))));

						success = CompareAndSet(to_check_2, location_check2, new_location_check2);
						if(success){ // step 14a1a
							location_check2 = new_location_check2;

							Slot new_location_target = (location & (~CHECK_1_FLAG_MASK));

							success = CompareAndSet(target, location, new_location_target);
							if(success){ // step 14a1a1
								location = new_location_target;
								
								target = to_swap;
								goto_14a1a1_step7 = true;

							} else { // step 14a1a2
							  	// bug fixed: change search_pos = to_check_2;
								//search_pos = pos;
								search_pos = to_check_2;
								goto_14a1a2_step6 = true;

							}

						} else { // step 14a1b
							// TODO: design fail
							printf("Insert key: %lu, step14a1b design fail\n", key);
						}

					} else { // step 14a2
						// TODO: design fail
						printf("Insert key: %lu, step14a2 design fail\n", key);
					}

				}

				location_check2 = __shfl(location_check2, lanid_target);
				goto_14a1a1_step7 = __shfl(goto_14a1a1_step7, lanid_target);
				goto_14a1a2_step6 = __shfl(goto_14a1a2_step6, lanid_target);

				if(goto_14a1a1_step7){
					target = __shfl(target, lanid_target);
					goto step_7;
				}

				if(goto_14a1a2_step6){
					search_pos = __shfl(search_pos, lanid_target);
					goto step_6;
				}

			} else { // step 14b

				bool goto_14b1a1a_step7 = false;

				if(tid == lanid_target){
					Slot new_location_swap = location_swap | SWAP_FLAG_MASK;

					success = CompareAndSet(to_swap, location_swap, new_location_swap);
					if(success){
						location_swap = new_location_swap;

						Slot new_location_target = (location & (~MAX_KEY)) | (location_swap & MAX_KEY); 
						new_location_target &= (~EMP_FLAG_MASK);
						new_location_target &= (~CHECK_1_FLAG_MASK);
						new_location_target &= (~BITMAP_MASK);

						success = CompareAndSet(target, location, new_location_target);
						if(success){ // step 14b1a
							location = new_location_target;

							new_location_swap = (location_swap & (~SWAP_FLAG_MASK)) | EMP_FLAG_MASK | CHECK_1_FLAG_MASK ;
							/*
							if(to_swap == 7468){
								printf("Insert key: %lu, step 14b1a add CHECK_1_FLAG_MASK\n", key);
							}*/

							success = CompareAndSet(to_swap, location_swap, new_location_swap);
							if(success){ // step 14b1a1

								Slot new_location_check2 = (location_check2 & (~CHECK_2_FLAG_MASK));
								new_location_check2 &= (~(BIT<<(BITMAP_SHIFT+(to_swap_offset))));

								success = CompareAndSet(to_check_2, location_check2, new_location_check2);			
								if(success){ // step 14b1a1a
									location_check2 = new_location_check2;
									target = to_swap;
									goto_14b1a1a_step7 = true;

								} else { // step 14b1a1b
									// TODO: design fail
									printf("Insert key: %lu, step14b1a1b design fail\n", key);
								}

							} else { // step 14b1a2
								// TODO: design fail
								printf("Insert key: %lu, step14b1a2 design fail\n", key);
							}

						} else { // step 14b1b
							// TODO: design fail
							printf("Insert key: %lu, step14b1b design fail\n", key);
						}

					}

				}

				goto_14b1a1a_step7 = __shfl(goto_14b1a1a_step7, lanid_target);

				if(goto_14b1a1a_step7){
					location_check2 = __shfl(location_check2, lanid_target);
					target = __shfl(target, lanid_target);
					goto step_7;
				}

			}

			swap_list &= (~(BIT<<to_swap_offset));
		}

		// step 13b
		bool goto_13b1_step10;
			
		if(tid == 0){
			Slot new_location_check2 = (location_check2 & (~CHECK_2_FLAG_MASK));
			new_location_check2 &= (~(BIT<<(BITMAP_SHIFT+(target-to_check_2))));

			success = CompareAndSet(to_check_2, location_check2, new_location_check2);
			if(success){
				location = new_location_check2;
				location_check2 = location;
				goto_13b1_step10 = true;

			} else { // step 13b2
				// TODO: design fail
				printf("Insert key: %lu, step13b2 design fail\n", key);
			}
		}

		location_check2 = __shfl(location_check2, 0);	
		goto_13b1_step10 = __shfl(goto_13b1_step10, 0);
		if(goto_13b1_step10){
			goto step_10;
		}

	}

}

__global__ void kernel(LL* items, LL* op, LL* result)
{
/*
	for(int op_id=0;op_id<NUM_ITEMS;op_id++){
		LL itm=items[op_id];
    	result[op_id] = WRONG_POS;
    	Slot location;

    	if(op_id == 3653){
    		printf("have done %d\n",op_id-1);
    	}

    	if(op_id == 2689){
    		printf("have done %d\n",op_id-1);
    	}

    	if(op[op_id]==ADD){
      		Insert(itm, &(result[op_id])); // return 1 or 0 or WRONG_POS(need rehash)
   	 	} else if(op[op_id]==DELETE){
      		Delete(itm, &(result[op_id])); // return 1 or 0
    	} else if(op[op_id]==SEARCH){
      		Find(itm, &(result[op_id]), &location); // return slot_no or WRONG_POS
    	}
	}
*/

	for(int i=0;i<FACTOR;i++){    		// FACTOR is the number of operations per thread
    	
    	int op_id=FACTOR*blockIdx.x+i;
    	if(op_id>=NUM_ITEMS) return;

    	// Grab the operation and the associated key and execute   		
    	LL itm=items[op_id];
    	result[op_id] = WRONG_POS;
    	Slot location;

    	if(op[op_id]==ADD){
      		Insert(itm, &(result[op_id])); // return 1 or 0 or WRONG_POS(need rehash)
   	 	} else if(op[op_id]==DELETE){
      		Delete(itm, &(result[op_id])); // return 1 or 0
    	} else if(op[op_id]==SEARCH){
      		Find(itm, &(result[op_id]), &location); // return slot_no or WRONG_POS
    	}
  	}
 
}

int main(int argc, char** argv)
{

	if (argc != 3) {
    	printf("Need two arguments: percent add ops and percent delete ops (e.g., 30 50 for 30%% add and 50%% delete).\nAborting...\n");
    	exit(1);
  	}

  	int adds=atoi(argv[1]);
  	int deletes=atoi(argv[2]);

   	if (adds+deletes > 100) {
    	printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     	exit(1);
  	}

	// Allocate hash table
	
	Slot slots[TABLE_SIZE];
	Slot * Cslots;

	int i;
	for(i=0;i<TABLE_SIZE;i++){
		slots[i] = EMP_FLAG_MASK;
	}

	#ifdef _CUTIL_H_
    	CUDA_SAFE_CALL(cudaMalloc((void**)&(Cslots), sizeof(Slot)*TABLE_SIZE ));
	#else
    	cudaMalloc((void**)&(Cslots), sizeof(Slot)*TABLE_SIZE );
	#endif

	#ifdef _CUTIL_H_
  		CUDA_SAFE_CALL(cudaMemcpy(Cslots, slots, sizeof(Slot)*TABLE_SIZE, cudaMemcpyHostToDevice));
	#else
  		cudaMemcpy(Cslots, slots, sizeof(Slot)*TABLE_SIZE, cudaMemcpyHostToDevice);
	#endif

	// Initialize the device memory
    init<<<1, THREADS_NUM>>>(Cslots);
  
  	LL op[NUM_ITEMS];		// Array of operations
  	LL items[NUM_ITEMS];		// Array of keys associated with operations
  	LL result[NUM_ITEMS];		// Array of outcomes
  	//LL expect_result[NUM_ITEMS]; // Array of expected result
/*
  	FILE * fp;
  	fp = fopen("/home/udms/Fanny/test/myfile_4.txt","r");
  	if(fp == NULL) exit(EXIT_FAILURE);

  	char line[100];
  	i=0;

  	while (fgets(line, 100, fp) != NULL)  {
  		char * p = strtok (line," "); 

		if(*p == 'I'){
			op[i]=ADD;
		} else if(*p == 'R'){
			op[i]=SEARCH;
		} else {
			op[i]=DELETE;
		}
		p = strtok(NULL," "); 

		if(*p == '0'){
			expect_result[i] = 0;
		} else {
			expect_result[i] = 1;
		}
		p = strtok(NULL," "); 
		
		unsigned long ul = strtoul (p, NULL, 0);
		items[i] = ul;

		i++;
	}

  	fclose(fp);
*/
  	srand(0);

  	// NUM_ITEMS is the total number of operations to execute
  	for(i=0;i<NUM_ITEMS;i++){
    	items[i]=10+rand()%KEYS;	// Keys
  	}

  	// Populate the op sequence
  	for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    	op[i]=ADD;
  	}
  	for(;i<(NUM_ITEMS*(adds+deletes))/100;i++){
    	op[i]=DELETE;
  	}
  	for(;i<NUM_ITEMS;i++){
    	op[i]=SEARCH;
  	}

  	//adds=(NUM_ITEMS*adds)/100;

  	// Allocate device memory

  	LL* Citems;
  	LL* Cop;
  	LL* Cresult;

	#ifdef _CUTIL_H_
  		CUDA_SAFE_CALL(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
  		CUDA_SAFE_CALL(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
  		CUDA_SAFE_CALL(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
  		CUDA_SAFE_CALL(cudaMemcpy(Citems,items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
  		CUDA_SAFE_CALL(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
	#else
  		cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS);
  		cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS);
  		cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS);
  		cudaMemcpy(Citems,items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
  		cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
	#endif

	// Calculate the number of thread blocks
  	// NUM_ITEMS = total number of operations to execute
  	// NUM_THREADS = number of threads per block
  	// FACTOR = number of operations per thread

  	//int blocks=(NUM_ITEMS%FACTOR==0)?(NUM_ITEMS/FACTOR):(NUM_ITEMS/FACTOR)+1;
  	int blocks=(NUM_ITEMS%(THREADS_NUM*FACTOR)==0)?NUM_ITEMS/(THREADS_NUM*FACTOR):(NUM_ITEMS/(THREADS_NUM*FACTOR))+1;

    // Launch main kernel

  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
  	cudaEventRecord(start, 0);
  
  	kernel<<<blocks, THREADS_NUM>>>(Citems, Cop, Cresult);
  
  	cudaEventRecord(stop, 0);
  	//cudaEventSynchronize(start);
  	cudaEventSynchronize(stop);
  	float time;
  	cudaEventElapsedTime(&time, start, stop);
  	cudaEventDestroy(start);
  	cudaEventDestroy(stop);

  	// Print kernel execution time in milliseconds

  	printf("%lf\n",time);

  	// Check for errors

  	cudaError_t error= cudaGetLastError();
  	if(cudaSuccess!=error){
    	printf("error:CUDA ERROR (%d) {%s}\n",error,cudaGetErrorString(error));
    	exit(-1);
  	}

  	// Move results back to host memory

	#ifdef _CUTIL_H_
  		CUDA_SAFE_CALL(cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost));
	#else
  		cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost);
	#endif
/*
	int insert_full = 0;
	int insert_fail = 0;
	int delete_fail = 0;
	int find_fail = 0;
	for(i=0;i<NUM_ITEMS;i++){
		if(op[i]==ADD){
			if(result[i]==WRONG_POS){
				if(i == 140700){
					printf("ADD full catch, line: %d, item: %d, result: %d, expect_result: %d\n", i, (unsigned long)items[i], (int)result[i], (int)expect_result[i]);
				}
				insert_full++;
			} else if(result[i] != expect_result[i]){
				printf("ADD fail, line: %d, item: %lu, result: %d, expect_result: %d\n", i, (unsigned long)items[i], (int)result[i], (int)expect_result[i]);
				insert_fail++;
			}
   	 	} else if(op[i]==DELETE){
      		if(result[i] != expect_result[i]){
      			//printf("DELETE fail, line: %d, item: %lu, result: %d, expect_result: %d\n", i, (unsigned long)items[i], (int)result[i], (int)expect_result[i]);
				delete_fail++;
			}
    	} else if(op[i]==SEARCH){
    		if(result[i]==WRONG_POS && expect_result[i]==0){
    			;
    		} else if(result[i]!=WRONG_POS && expect_result[i]==1){
    			;
    		} else {
    			printf("SEARCH fail, line: %d, item: %lu, result: %d, expect_result: %d\n", i, (unsigned long)items[i], (int)result[i], (int)expect_result[i]);
    			find_fail++;
    		}
    	} 
	}

	printf("insert_full: %d insert_fail: %d delete_fail: %d find_fail: %d\n", insert_full, insert_fail, delete_fail, find_fail); */

	return 0;
}
